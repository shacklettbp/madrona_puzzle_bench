import torch
from torch import nn
import torch.nn.functional as F
import torch._dynamo
from torch import optim
from torch.func import vmap
import os
from os import environ as env_vars
from typing import Callable
from dataclasses import dataclass
from typing import List, Optional, Dict
from .profile import profile
from time import time
from pathlib import Path
import os

from .cfg import TrainConfig, SimInterface
from .rollouts import RolloutManager, Rollouts
from .amp import AMPState
from .actor_critic import ActorCritic
from .moving_avg import EMANormalizer
from .learning_state import LearningState
from .replay_buffer import NStepReplay

import datetime

@dataclass(frozen = True)
class MiniBatch:
    obs: List[torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    values_intrinsic: torch.Tensor
    advantages: torch.Tensor
    advantages_intrinsic: torch.Tensor
    rnn_start_states: tuple[torch.Tensor, ...]


@dataclass
class PPOStats:
    loss : float = 0
    action_loss : float = 0
    value_loss : float = 0
    value_loss_intrinsic : float = 0
    intrinsic_loss: float = 0
    entropy_loss : float = 0
    returns_mean : float = 0
    returns_stddev : float = 0
    value_loss_array : torch.Tensor = None
    intrinsic_reward_array : torch.Tensor = None
    entropy_loss_array : torch.Tensor = None

@dataclass(frozen = True)
class UpdateResult:
    obs: torch.Tensor
    actions : torch.Tensor
    rewards : torch.Tensor
    returns : torch.Tensor
    dones: torch.Tensor
    values : torch.Tensor
    advantages : torch.Tensor
    bootstrap_values : torch.Tensor
    ppo_stats : PPOStats


def _mb_slice(tensor, inds):
    # Tensors come from the rollout manager as (C, T, N, ...)
    # Want to select mb from C * N and keep sequences of length T

    return tensor.transpose(0, 1).reshape(
        tensor.shape[1], tensor.shape[0] * tensor.shape[2], *tensor.shape[3:])[:, inds, ...]

def _mb_slice_rnn(rnn_state, inds):
    # RNN state comes from the rollout manager as (C, :, :, N, :)
    # Want to select minibatch from C * N and keep sequences of length T

    reshaped = rnn_state.permute(1, 2, 0, 3, 4).reshape(
        rnn_state.shape[1], rnn_state.shape[2], -1, rnn_state.shape[4])

    return reshaped[:, :, inds, :] 

def _gather_minibatch(rollouts : Rollouts,
                      advantages : torch.Tensor,
                      advantages_intrinsic : torch.Tensor,
                      inds : torch.Tensor,
                      amp : AMPState):
    obs_slice = tuple(_mb_slice(obs, inds) for obs in rollouts.obs)

    # Print if in third room
    #third_room_count = (obs_slice[0][:,3] > 0.65).sum()
    #if third_room_count > 0:
    #    print("There are ", third_room_count, "agents in the third room")
    
    actions_slice = _mb_slice(rollouts.actions, inds)
    log_probs_slice = _mb_slice(rollouts.log_probs, inds).to(
        dtype=amp.compute_dtype)
    dones_slice = _mb_slice(rollouts.dones, inds)
    rewards_slice = _mb_slice(rollouts.rewards, inds).to(
        dtype=amp.compute_dtype)
    values_slice = _mb_slice(rollouts.values, inds).to(
        dtype=amp.compute_dtype)
    advantages_slice = _mb_slice(advantages, inds).to(
        dtype=amp.compute_dtype)
    if rollouts.values_intrinsic is not None:
        values_intrinsic_slice = _mb_slice(rollouts.values_intrinsic, inds).to(
            dtype=amp.compute_dtype)
        advantages_intrinsic_slice = _mb_slice(advantages_intrinsic, inds).to(
            dtype=amp.compute_dtype)
    else:
        values_intrinsic_slice = None
        advantages_intrinsic_slice = None

    rnn_starts_slice = tuple(
        _mb_slice_rnn(state, inds) for state in rollouts.rnn_start_states)

    return MiniBatch(
        obs=obs_slice,
        actions=actions_slice,
        log_probs=log_probs_slice,
        dones=dones_slice,
        rewards=rewards_slice,
        values=values_slice,
        values_intrinsic=values_intrinsic_slice,
        advantages=advantages_slice,
        advantages_intrinsic=advantages_intrinsic_slice,
        rnn_start_states=rnn_starts_slice,
    )

def _compute_advantages(cfg : TrainConfig,
                        amp : AMPState,
                        advantages_out : torch.Tensor,
                        advantages_intrinsic_out : torch.Tensor,
                        rollouts : Rollouts):
    # This function is going to be operating in fp16 mode completely
    # when mixed precision is enabled since amp.compute_dtype is fp16
    # even though there is no autocast here. Unclear if this is desirable or
    # even beneficial for performance.

    num_chunks, steps_per_chunk, N = rollouts.dones.shape[0:3]
    T = num_chunks * steps_per_chunk

    seq_dones = rollouts.dones.view(T, N, 1)
    seq_rewards = rollouts.rewards.view(T, N, 1)
    seq_values = rollouts.values.view(T, N, 1)
    seq_advantages_out = advantages_out.view(T, N, 1)

    next_advantage = 0.0
    next_values = rollouts.bootstrap_values
    for i in reversed(range(cfg.steps_per_update)):
        cur_dones = seq_dones[i].to(dtype=amp.compute_dtype)
        cur_rewards = seq_rewards[i].to(dtype=amp.compute_dtype)
        cur_values = seq_values[i].to(dtype=amp.compute_dtype)

        next_valid = 1.0 - cur_dones

        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        if cfg.ppo.no_advantages:
            td_err = (cur_rewards + 
                cfg.gamma * next_valid * next_values) # Don't subtract off cur_values
        else:
            td_err = (cur_rewards + 
                cfg.gamma * next_valid * next_values - cur_values)

        # A_t = sum (gamma * lambda)^(l - 1) * delta_l (EQ 16 GAE)
        #     = delta_t + gamma * lambda * A_t+1
        cur_advantage = (td_err +
            cfg.gamma * cfg.gae_lambda * next_valid * next_advantage)

        seq_advantages_out[i] = cur_advantage

        next_advantage = cur_advantage
        next_values = cur_values
    
    # Repeat for intrinsic values and advantages if not None
    if rollouts.values_intrinsic is not None and cfg.ppo.use_intrinsic_loss:
        seq_rewards_intrinsic = rollouts.rewards_intrinsic.view(T, N, 1)
        seq_values_intrinsic = rollouts.values_intrinsic.view(T, N, 1)
        seq_advantages_intrinsic_out = advantages_intrinsic_out.view(T, N, 1)

        next_advantage = 0.0
        next_values = rollouts.bootstrap_values_intrinsic
        for i in reversed(range(cfg.steps_per_update)):
            cur_dones = seq_dones[i].to(dtype=amp.compute_dtype)
            cur_rewards_intrinsic = seq_rewards_intrinsic[i].to(dtype=amp.compute_dtype)
            cur_values = seq_values_intrinsic[i].to(dtype=amp.compute_dtype)

            #next_valid = 1.0 - cur_dones

            # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            if cfg.ppo.no_advantages:
                td_err = (cur_rewards_intrinsic + 
                    #cfg.gamma * next_valid * next_values)
                    cfg.gamma * next_values)
            else:
                td_err = (cur_rewards_intrinsic + 
                    #cfg.gamma * next_valid * next_values - cur_values)
                    cfg.gamma * next_values - cur_values)
            
            # A_t = sum (gamma * lambda)^(l - 1) * delta_l (EQ 16 GAE)
            #     = delta_t + gamma * lambda * A_t+1
            cur_advantage = (td_err +
                cfg.gamma * cfg.gae_lambda * next_valid * next_advantage)
            
            seq_advantages_intrinsic_out[i] = cur_advantage

            next_advantage = cur_advantage
            next_values = cur_values

def _compute_action_scores(cfg, amp, advantages):
    if not cfg.normalize_advantages:
        return advantages
    else:
        # Unclear from docs if var_mean is safe under autocast
        with amp.disable():
            var, mean = torch.var_mean(advantages.to(dtype=torch.float32))
            action_scores = advantages - mean
            action_scores.mul_(torch.rsqrt(var.clamp(min=1e-5)))

            return action_scores.to(dtype=amp.compute_dtype)

def _ppo_update(cfg : TrainConfig,
                amp : AMPState,
                mb : MiniBatch,
                actor_critic : ActorCritic,
                optimizer : torch.optim.Optimizer,
                value_normalizer : EMANormalizer,
                value_normalizer_intrinsic : EMANormalizer,
                world_weights: torch.Tensor,
            ):
    with amp.enable():
        with profile('AC Forward', gpu=True):
            new_log_probs, entropies, new_values, new_values_intrinsic, reward_intrinsic = actor_critic.fwd_update(
                mb.rnn_start_states, mb.dones, mb.actions, *mb.obs)

        #print("advantages", mb.advantages.shape, mb.advantages_intrinsic.shape)
        with torch.no_grad():
            if cfg.ppo.use_intrinsic_loss:
                action_scores = _compute_action_scores(cfg, amp, mb.advantages + mb.advantages_intrinsic)
            else:
                action_scores = _compute_action_scores(cfg, amp, mb.advantages)

        #if mb.dones.sum() > 0: # VISHNU LOGGING
        #    print("We have a done!")

        ratio = torch.exp(new_log_probs - mb.log_probs)
        surr1 = action_scores * ratio
        surr2 = action_scores * (
            torch.clamp(ratio, 1.0 - cfg.ppo.clip_coef, 1.0 + cfg.ppo.clip_coef))

        action_obj = torch.min(surr1, surr2)

        returns = mb.advantages + mb.values
        if cfg.ppo.use_intrinsic_loss:
            intrinsic_returns = mb.advantages_intrinsic + mb.values_intrinsic
        #else:
        #    intrinsic_returns = torch.zeros_like(returns)

        if cfg.ppo.clip_value_loss:
            with torch.no_grad():
                low = mb.values - cfg.ppo.clip_coef
                high = mb.values + cfg.ppo.clip_coef

            new_values = torch.clamp(new_values, low, high)

            # Might as well clip the intrinsic values too? 
            if cfg.ppo.use_intrinsic_loss:
                with torch.no_grad():
                    low = mb.values_intrinsic - cfg.ppo.clip_coef
                    high = mb.values_intrinsic + cfg.ppo.clip_coef

                new_values_intrinsic = torch.clamp(new_values_intrinsic, low, high)

        normalized_returns = value_normalizer(amp, returns)
        #print("Value shapes", new_values.shape, normalized_returns.shape, new_values_intrinsic.shape, normalized_returns_intrinsic.shape)
        value_loss_array = 0.5 * F.mse_loss(
            new_values, normalized_returns, reduction='none')
        if cfg.ppo.use_intrinsic_loss:
            normalized_returns_intrinsic = value_normalizer_intrinsic(amp, intrinsic_returns)
            value_loss_intrinsic = 0.5 * F.mse_loss(
                new_values_intrinsic, normalized_returns_intrinsic, reduction='none')

        '''
        action_obj = torch.mean(action_obj)
        value_loss = torch.mean(value_loss_array)
        entropies = torch.mean(entropies)
        '''
        # Weighted means by world_weights
        #print("Weight shapes")
        #print(action_obj.shape, world_weights.shape, value_loss_array.shape, entropies.shape)
        world_weights = world_weights[None, :, None]
        action_obj = (torch.sum(action_obj * world_weights) / world_weights.sum()).mean()
        value_loss = (torch.sum(value_loss_array * world_weights) / world_weights.sum()).mean()
        entropy_loss_array = torch.clone(entropies).detach()
        entropies = (torch.sum(entropies * world_weights) / world_weights.sum()).mean()

        if cfg.ppo.use_intrinsic_loss:
            '''
            value_loss_intrinsic = torch.mean(value_loss_intrinsic)
            intrinsic_loss = torch.mean(reward_intrinsic)
            '''
            value_loss_intrinsic = torch.sum(value_loss_intrinsic * world_weights) / world_weights.sum()
            intrinsic_loss = torch.sum(reward_intrinsic * world_weights) / world_weights.sum()
            loss = (
                - action_obj # Maximize the action objective function
                + cfg.ppo.value_loss_coef * value_loss
                + cfg.ppo.value_loss_intrinsic_coef * value_loss_intrinsic
                + cfg.ppo.intrinsic_loss_coef * intrinsic_loss # Minimize intrinsic loss
                - cfg.ppo.entropy_coef * entropies # Maximize entropy
            )
        else:
            value_loss_intrinsic = torch.tensor(0.0)
            intrinsic_loss = torch.tensor(0.0)
            loss = (
                - action_obj # Maximize the action objective function
                + cfg.ppo.value_loss_coef * value_loss
                - cfg.ppo.entropy_coef * entropies # Maximize entropy
            )

    with profile('Optimize'):
        if amp.scaler is None:
            loss.backward()
            #print("MAX")
            #for k, v in actor_critic.named_parameters():
            #    print("  ", k, torch.max(v.grad))
            #print("MIN")
            #for k, v in actor_critic.named_parameters():
            #    print("  ", k, torch.min(v.grad))
            #print("MEAN")
            #for k, v in actor_critic.named_parameters():
            #    print("  ", k, torch.mean(v.grad))

            nn.utils.clip_grad_norm_(
                actor_critic.parameters(), cfg.ppo.max_grad_norm)
            optimizer.step()
        else:
            amp.scaler.scale(loss).backward()
            amp.scaler.unscale_(optimizer)
            #print("MAX")
            #for k, v in actor_critic.named_parameters():
            #    print("  ", k, torch.max(v.grad))
            #print("MIN")
            #for k, v in actor_critic.named_parameters():
            #    print("  ", k, torch.min(v.grad))
            #print("MEAN")
            #for k, v in actor_critic.named_parameters():
            #    print("  ", k, torch.mean(v.grad))
            nn.utils.clip_grad_norm_(
                actor_critic.parameters(), cfg.ppo.max_grad_norm)
            amp.scaler.step(optimizer)
            amp.scaler.update()

        optimizer.zero_grad()

    #print("Intrinsic reward array shape", reward_intrinsic.shape)
    #print(reward_intrinsic)

    with torch.no_grad():
        returns_var, returns_mean = torch.var_mean(normalized_returns)
        returns_stddev = torch.sqrt(returns_var)

        stats = PPOStats(
            loss = loss.cpu().float().item(),
            action_loss = -(action_obj.cpu().float().item()),
            value_loss = (cfg.ppo.value_loss_coef * value_loss.cpu().float().item()),
            value_loss_intrinsic = (cfg.ppo.value_loss_intrinsic_coef * value_loss_intrinsic.cpu().float().item()),
            intrinsic_loss = (cfg.ppo.intrinsic_loss_coef * intrinsic_loss.cpu().float().item()),
            entropy_loss = -(cfg.ppo.entropy_coef * entropies.cpu().float().item()),
            returns_mean = returns_mean.cpu().float().item(),
            returns_stddev = returns_stddev.cpu().float().item(),
            value_loss_array = value_loss_array,#[-1,:,0], # Collapse to one per checkpoint
            intrinsic_reward_array = reward_intrinsic if reward_intrinsic is not None else None, # Collapse to one per checkpoint
            entropy_loss_array = entropy_loss_array,
        )

    return stats

def weighted_stratified_sample_corrected(data, labels, total_samples, class_weights):
    # Calculate class distribution and adjust by class weights
    print("Labels", torch.unique(labels), class_weights.shape[0], labels.shape)
    counts = torch.bincount(labels, minlength=class_weights.shape[0])
    adjusted_counts = counts.float() * class_weights
    
    # Calculate the proportion of each class after adjustment
    adjusted_proportions = adjusted_counts / adjusted_counts.sum()
    
    # Compute initial samples per class without rounding
    float_samples_per_class = adjusted_proportions * total_samples
    
    # Determine rounded and residual values
    samples_per_class = float_samples_per_class.floor().long()
    residuals = float_samples_per_class - samples_per_class.float()
    
    # Correct the total number of samples to match exactly
    while samples_per_class.sum() < total_samples:
        idx = residuals.argmax()
        samples_per_class[idx] += 1
        residuals[idx] = 0  # Prevent this class from being selected again

    print("Samples per class", samples_per_class)
    print("Adjusted counts", adjusted_counts)

    sampled_data = []
    sampled_labels = []

    for class_id in range(class_weights.shape[0]):
        class_data = data[labels == class_id]
        num_samples_class = samples_per_class[class_id]

        # If the class is empty or no samples needed, skip
        if len(class_data) == 0 or num_samples_class == 0:
            continue

        # Select indices randomly for the current class
        #print("Class data", class_data.shape, num_samples_class)
        if num_samples_class > len(class_data):
            indices = torch.randperm(len(class_data)).repeat(num_samples_class // len(class_data) + 1)[:num_samples_class]
        else:
            indices = torch.randperm(len(class_data))[:num_samples_class]

        sampled_data.append(class_data[indices])
        sampled_labels.append(labels[labels == class_id][indices])

    # Concatenate sampled data and labels from all classes
    sampled_data = torch.cat(sampled_data, dim=0)
    sampled_labels = torch.cat(sampled_labels, dim=0)

    #print("Sampled data shape", sampled_data.shape)

    return sampled_data, sampled_labels

def make_non_decreasing_custom(tensor, axis=0):
    # Clone the tensor to avoid modifying the original one
    result_tensor = tensor.clone()
    # Apply cumulative maximum along the specified axis
    for idx in range(1, tensor.shape[axis]):
        # Take slices and compare with previous ones
        previous_slice = result_tensor.select(axis, idx-1)
        current_slice = result_tensor.select(axis, idx)
        # Update the current slice based on the condition
        result_tensor.select(axis, idx).copy_(torch.max(previous_slice, current_slice))
    return result_tensor

def _update_iter(cfg : TrainConfig,
                 amp : AMPState,
                 num_train_seqs : int,
                 sim : SimInterface,
                 rollout_mgr : RolloutManager,
                 advantages : torch.Tensor,
                 advantages_intrinsic: torch.Tensor,
                 actor_critic : ActorCritic,
                 optimizer : torch.optim.Optimizer,
                 scheduler : torch.optim.lr_scheduler.LRScheduler,
                 value_normalizer : EMANormalizer,
                 value_normalizer_intrinsic : EMANormalizer,
                 replay_buffer: NStepReplay,
                 user_cb,
            ):
    with torch.no_grad():
        actor_critic.eval()
        value_normalizer.eval()
        # This is where the simulator loop happens that executes the TaskGraph.
        with profile('Collect Rollouts'):
            world_weights = torch.ones(user_cb.num_worlds, device=advantages.device)
            if type(user_cb).__name__ == "GoExplore":
                # Let's also do some checkpoint restores here
                # This should actually happen before collecting rollouts! But doesn't matter beyond first step i think
                if user_cb.new_frac > 0.002 and not user_cb.choose_worlds:
                    desired_samples = int(user_cb.num_worlds*user_cb.new_frac)
                    user_cb.checkpoint_resets[:, 0] = 1
                    if user_cb.importance_test:
                        # Only copy from first half of fixed worlds
                        source_worlds = user_cb.checkpoints[(desired_samples + user_cb.num_worlds) // 2:]
                        #print("Source worlds", source_worlds.shape, desired_samples, user_cb.num_worlds)
                        num_copies = ((user_cb.num_worlds // (user_cb.num_worlds - desired_samples)) - 1)*2
                        user_cb.checkpoints[:desired_samples] = source_worlds.repeat(num_copies, 1)
                        if user_cb.importance_weights:
                            # Higher weight on the worlds not copied
                            world_weights[desired_samples:(desired_samples + user_cb.num_worlds) // 2] = num_copies + 1
                            #print(torch.unique(world_weights, return_counts=True))
                            # Renormalize to average weight 1
                            world_weights /= world_weights.mean()
                            #print(torch.unique(world_weights, return_counts=True))
                    elif user_cb.intelligent_sample == "bin":
                        # First bin the source worlds
                        source_bins = user_cb.map_states_to_bins(user_cb.obs, max_bin=False)[0,desired_samples:]
                        print("Bins shape", source_bins.shape)
                        # Now we divide by 200 to get the higher-level state in the progression
                        source_bins = source_bins // 200
                        # Update intelligent_weights if we are using td_weights
                        sampling_weights = torch.tensor(user_cb.intelligent_weights, device = user_cb.checkpoints.device)
                        print("pre mod sampling weights", sampling_weights)
                        if user_cb.success_weights:
                            print("Running success rate", user_cb.running_success_rate)
                            # Exponentiate negative running success rate
                            sampling_weights = torch.exp(-user_cb.running_success_rate * user_cb.success_rate_temp)
                            # Normalize
                            sampling_weights /= sampling_weights.sum()
                            #sampling_weights = user_cb.running_td_error / user_cb.running_td_error.sum()
                        else:
                            # Do count-based weights
                            sampling_weights = torch.bincount(source_bins.to(user_cb.checkpoints.device), minlength=sampling_weights.shape[0])
                            sampling_weights = 1. / (torch.pow(sampling_weights, 0.5) + 1)
                            sampling_weights /= sampling_weights.sum()
                        print("Sampling weights", sampling_weights)
                        print("Bin counts", torch.unique(source_bins, return_counts=True))
                        # Now we need to weight the sampling to sample less from source_bins == 0
                        sampled_checkpoints, _ = weighted_stratified_sample_corrected(user_cb.checkpoints[desired_samples:], source_bins.to(user_cb.checkpoints.device), desired_samples, sampling_weights)
                        user_cb.checkpoints[:desired_samples] = sampled_checkpoints
                        # Let's not re-weight this for now
                    elif user_cb.intelligent_sample == "value":
                        world_values = rollout_mgr.bootstrap_values[desired_samples:][:,0]
                        print("World values", torch.mean(world_values), torch.var(world_values))
                        # Now we convert these into "success fracs" by clipping to 13, then dividing by 13
                        world_values = torch.clamp(world_values, 0, 13) / 13
                        sampling_weights = torch.exp(-world_values * user_cb.success_rate_temp)
                        # Now sample from this
                        print("Sampling weights", sampling_weights.shape)
                        sample_rows = torch.multinomial(sampling_weights, desired_samples, replacement=True)
                        user_cb.checkpoints[:desired_samples] = torch.clone(user_cb.checkpoints[desired_samples:][sample_rows])
                    else:
                        user_cb.checkpoints[:desired_samples] = user_cb.checkpoints[desired_samples:].repeat((user_cb.num_worlds // (user_cb.num_worlds - desired_samples)) - 1, 1)
                    user_cb.worlds.step()

            prev_bins = torch.clone(rollout_mgr.obs[2][0][-1])

            rollouts = rollout_mgr.collect(amp, sim, actor_critic, value_normalizer, value_normalizer_intrinsic)
            #print("Testing: adding to buffer")
            #replay_buffer.add_to_buffer(rollouts)
            #print("Testing: load oldest thing in buffer")
            #rollouts = replay_buffer.get_last(rollouts)
            #print("Testing: load multiple from buffer")
            #rollouts = replay_buffer.get_multiple(rollouts)

            # Now modify the rewards in the rollouts by adding reward when closer to "exit"
            if type(user_cb).__name__ == "GoExplore":
                if user_cb.reward_type == "count":
                    # Get raw bin counts
                    all_bins = user_cb.map_states_to_bins(rollouts.obs)
                    bin_counts = user_cb.bin_count[all_bins]
                    # Make reward inversely proportional to bin count
                    reward_bonus = 1.0 / (torch.sqrt(bin_counts) + 1) #[...,None].repeat(1,1,2).view(bin_counts.shape[0],-1,1) + 1)
                    # Make the mean reward bonus be bin_reward_boost
                    mean_reward_bonus = reward_bonus.mean()
                    # Check that mean_reward_bonus is not zero
                    reward_bonus *= user_cb.bin_reward_boost / mean_reward_bonus
                    # Add the reward bonus to the rollouts
                    rollouts.rewards.view(-1, *rollouts.rewards.shape[2:])[:] += reward_bonus.view(reward_bonus.shape[0],-1,1)
                elif user_cb.reward_type == "distance":
                    # Compute change in exit dist
                    all_bins = user_cb.map_states_to_bins(rollouts.obs) # num_timesteps * num_worlds
                    #if user_cb.max_progress < 1.01:
                    reward_bonus_1 = user_cb.start_bin_steps[all_bins].float()
                    #print("Reward bonus", reward_bonus_1)
                    mean_reward_bonus = reward_bonus_1.mean()
                    reward_bonus_1 *= user_cb.bin_reward_boost / mean_reward_bonus
                    #print("Normalized reward bonus", reward_bonus_1)
                    #print(reward_bonus_1.sum(axis=0))
                    #rollouts.rewards.view(-1, *rollouts.rewards.shape[2:])[:] *= 0
                    rollouts.rewards.view(-1, *rollouts.rewards.shape[2:])[:] += reward_bonus_1.view(reward_bonus_1.shape[0],-1,1) #* user_cb.bin_reward_boost * 0.5 #[...,None].repeat(1,1,2).view(reward_bonus_1.shape[0],-1,1) #* user_cb.bin_reward_boost * 0.5
                    '''
                    max_bin_steps = 200
                    if user_cb.bin_steps[user_cb.bin_steps < 200].size(dim=0) > 0:
                        max_bin_steps = user_cb.bin_steps[user_cb.bin_steps < 200].max()
                    reward_bonus_2 = max_bin_steps - user_cb.bin_steps[all_bins]
                    reward_bonus_2[reward_bonus_2 < 0] = 0
                    '''
                    #reward_bonus = (user_cb.bin_steps[all_bins[1:]] < user_cb.bin_steps[all_bins[:-1]]).float() - (user_cb.bin_steps[all_bins[1:]] > user_cb.bin_steps[all_bins[:-1]]).float()
                    #rollouts.rewards.view(-1, *rollouts.rewards.shape[2:])[:-1] += reward_bonus[...,None].repeat(1,1,2).view(reward_bonus.shape[0],-1,1) * user_cb.bin_reward_boost
                    #rollouts.rewards.view(-1, *rollouts.rewards.shape[2:])[:] += reward_bonus_2[...,None].repeat(1,1,2).view(reward_bonus_2.shape[0],-1,1) * user_cb.bin_reward_boost
                elif user_cb.reward_type == "subtask":
                    # Let's extract subtask transitions from the rollouts
                    # We want progress from subtask i to i+1 to be associated with positive reward
                    # but handle the case where we fall back to the previous subtask when we mess up
                    # We can do this by checking if the subtask stays at i+1 for a few steps? Or by providing negative reward for backward progress
                    # Let's do negative reward for backward progress for now
                    all_bins = user_cb.map_states_to_bins(rollouts.obs, max_bin=False) # num_timesteps * num_worlds
                    all_bins = all_bins // 200
                    # Copy from previous rollout if not steps_remaining = 200
                    print("All bins shape", all_bins[0].shape)
                    print("Prev bins shape", prev_bins.shape)
                    print("Filter shape", prev_bins[rollouts.obs[9][0,0,:,0] < 195][:,0].shape) # Should be 200 not 195
                    #all_bins[0][rollouts.obs[9][0,0,:,0] < 195] = torch.maximum(all_bins[0][rollouts.obs[9][0,0,:,0] < 195], prev_bins[rollouts.obs[9][0,0,:,0] < 195][:,0])

                    # If there are any "dones", add 4 to bin_id so that monotonicity doesn't mess with this
                    print("Dones shape", rollouts.dones.shape)
                    non_decreasing_dones = torch.clone(rollouts.dones.view(-1, *rollouts.dones.shape[2:3])).int()
                    non_decreasing_dones[1:] += torch.cummax(non_decreasing_dones[:-1], dim=0)[0]
                    print("dones unique", torch.unique(non_decreasing_dones, return_counts=True))
                    print("All bins shape", all_bins.shape)
                    #all_bins += 4*non_decreasing_dones
                    # Print unique bins
                    print("Unique bins", torch.unique(all_bins, return_counts=True))
                    # Also stick this into level type
                    rollouts.obs[2][0][:,:,0] = all_bins % 4 # torch.randint(4, rollouts.obs[2][0][:,:,0].shape) # all_bins % 4
                    # Make the bins monotonically increasing
                    all_bins = make_non_decreasing_custom(all_bins, axis=0)
                    print("Unique bins", torch.unique(all_bins, return_counts=True))
                    # Also stick this into level type
                    #rollouts.obs[2][0][:,:,0] = all_bins % 4 # torch.randint(4, rollouts.obs[2][0][:,:,0].shape) # all_bins % 4
                    # Now compute where the subtask transitions are
                    positive_transitions = (all_bins[1:] == all_bins[:-1] + 1)
                    # Print transition count
                    print("Positive transitions", torch.unique(positive_transitions, return_counts=True))
                    # Reward bonus from positive transitions
                    reward_bonus = positive_transitions.float()*10.0
                    # Also set dones to 1 if we have a positive transition
                    #rollouts.dones.view(-1, *rollouts.dones.shape[2:])[:-1] += positive_transitions[...,None] # I think this is [:-1] for dones, not [1:]. Since the reward and done are both corresponding to the state-action
                    print("Rollout shape", rollouts.rewards.shape)
                    print("reward bonus shape", reward_bonus.shape)
                    rollouts.rewards.view(-1, *rollouts.rewards.shape[2:])[:-1] += reward_bonus[...,None]
                    # Need to also alter "done" signal and timestep...
                elif user_cb.reward_type == "none":
                    pass
                else:
                    raise Exception("Invalid reward type")

        # Dump the rollout
        '''
        curr_rand = torch.rand((1,))[0]
        if curr_rand < 0.05:
            # Dump the features
            now = datetime.datetime.now()
            dir_path = "/data/rl/madrona_3d_example/data_dump/" + cfg.run_name + "/"
            isExist = os.path.exists(dir_path)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(dir_path)
            torch.save(rollouts, dir_path + str(now) + ".pt")
        '''
    
        # Engstrom et al suggest recomputing advantages after every epoch
        # but that's pretty annoying for a recurrent policy since values
        # need to be recomputed. https://arxiv.org/abs/2005.12729
        with profile('Compute Advantages'):
            _compute_advantages(cfg,
                                amp,
                                advantages,
                                advantages_intrinsic,
                                rollouts)
            #print("Advantages", advantages.shape, advantages_intrinsic.shape)
    
    actor_critic.train()
    value_normalizer.train()

    with profile('PPO'):
        aggregate_stats = PPOStats()
        num_stats = 0

        for epoch in range(cfg.ppo.num_epochs):
            for inds in torch.arange(num_train_seqs).chunk(
            #for inds in torch.randperm(num_train_seqs).chunk( # VISHNU: ASSUME ONE MINIBATCH, randperm messes up tracking
                    cfg.ppo.num_mini_batches):
                with torch.no_grad(), profile('Gather Minibatch', gpu=True):
                    mb = _gather_minibatch(rollouts, advantages, advantages_intrinsic, inds, amp)
                cur_stats = _ppo_update(cfg,
                                        amp,
                                        mb,
                                        actor_critic,
                                        optimizer,
                                        value_normalizer,
                                        value_normalizer_intrinsic,
                                        world_weights)

                with torch.no_grad():
                    num_stats += 1
                    aggregate_stats.loss += (cur_stats.loss - aggregate_stats.loss) / num_stats
                    aggregate_stats.action_loss += (
                        cur_stats.action_loss - aggregate_stats.action_loss) / num_stats
                    aggregate_stats.value_loss += (
                        cur_stats.value_loss - aggregate_stats.value_loss) / num_stats
                    aggregate_stats.value_loss_intrinsic += (
                        cur_stats.value_loss_intrinsic - aggregate_stats.value_loss_intrinsic) / num_stats
                    aggregate_stats.intrinsic_loss += (
                        cur_stats.intrinsic_loss - aggregate_stats.intrinsic_loss) / num_stats
                    aggregate_stats.entropy_loss += (
                        cur_stats.entropy_loss - aggregate_stats.entropy_loss) / num_stats
                    aggregate_stats.returns_mean += (
                        cur_stats.returns_mean - aggregate_stats.returns_mean) / num_stats
                    # FIXME
                    aggregate_stats.returns_stddev += (
                        cur_stats.returns_stddev - aggregate_stats.returns_stddev) / num_stats
                    aggregate_stats.value_loss_array = cur_stats.value_loss_array
                    #aggregate_stats.intrinsic_reward_array = cur_stats.intrinsic_reward_array
                    aggregate_stats.entropy_loss_array = cur_stats.entropy_loss_array
    #aggregate_stats.intrinsic_reward_array = rollouts.rewards_intrinsic.view(-1, *rollouts.rewards_intrinsic.shape[2:])[-1,:,0].detach().float() if rollouts.rewards_intrinsic is not None else None # Get the intrinsic reward of the last step to weight the checkpoint
    aggregate_stats.intrinsic_reward_array = rollouts.rewards_intrinsic.view(-1, *rollouts.rewards_intrinsic.shape[2:]).detach().float() if rollouts.rewards_intrinsic is not None else None

    return UpdateResult(
        obs = rollouts.obs,
        actions = rollouts.actions.view(-1, *rollouts.actions.shape[2:]),
        rewards = rollouts.rewards.view(-1, *rollouts.rewards.shape[2:]),
        returns = rollouts.returns.view(-1, *rollouts.returns.shape[2:]),
        dones = rollouts.dones.view(-1, *rollouts.dones.shape[2:]),
        values = rollouts.values.view(-1, *rollouts.values.shape[2:]),
        advantages = advantages.view(-1, *advantages.shape[2:]),
        bootstrap_values = rollouts.bootstrap_values,
        ppo_stats = aggregate_stats,
    )

def _update_loop(update_iter_fn : Callable,
                 gpu_sync_fn : Callable,
                 user_cb : Callable,
                 cfg : TrainConfig,
                 num_agents: int,
                 sim : SimInterface,
                 rollout_mgr : RolloutManager,
                 learning_state : LearningState,
                 start_update_idx : int,
                 replay_buffer: NStepReplay):
    num_train_seqs = num_agents * cfg.num_bptt_chunks
    assert(num_train_seqs % cfg.ppo.num_mini_batches == 0)

    advantages = torch.zeros_like(rollout_mgr.rewards)
    advantages_intrinsic = torch.zeros_like(rollout_mgr.rewards)

    for update_idx in range(start_update_idx, cfg.num_updates):
        update_start_time  = time()

        with profile("Update Iter Timing"):
            update_result = update_iter_fn(
                cfg,
                learning_state.amp,
                num_train_seqs,
                sim,
                rollout_mgr,
                advantages,
                advantages_intrinsic,
                learning_state.policy,
                learning_state.optimizer,
                learning_state.scheduler,
                learning_state.value_normalizer,
                learning_state.value_normalizer_intrinsic,
                replay_buffer,
                user_cb,
            )

            gpu_sync_fn()

        profile.gpu_measure()
        profile.commit()

        update_end_time = time()
        update_time = update_end_time - update_start_time
        user_cb(update_idx, update_time, update_result, learning_state)

def train(dev, sim, cfg, actor_critic, update_cb, restore_ckpt=None):
    print(cfg)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    num_agents = sim.actions.shape[0]

    actor_critic = actor_critic.to(dev)

    optimizer = optim.Adam(actor_critic.parameters(), lr=cfg.lr)

    amp = AMPState(dev, cfg.mixed_precision)

    value_normalizer = EMANormalizer(cfg.value_normalizer_decay,
                                     disable=not cfg.normalize_values)
    value_normalizer = value_normalizer.to(dev)

    value_normalizer_intrinsic = EMANormalizer(cfg.value_normalizer_decay,
                                        disable=not cfg.normalize_values)
    value_normalizer_intrinsic = value_normalizer_intrinsic.to(dev)

    learning_state = LearningState(
        policy = actor_critic,
        optimizer = optimizer,
        scheduler = None,
        value_normalizer = value_normalizer,
        value_normalizer_intrinsic = value_normalizer_intrinsic,
        amp = amp,
    )

    # Restore a previous policy, nothing to do with the state of the world.
    if restore_ckpt != None:
        start_update_idx = learning_state.load(restore_ckpt)
    else:
        start_update_idx = 0

    rollout_mgr = RolloutManager(dev, sim, cfg.steps_per_update,
        cfg.num_bptt_chunks, amp, actor_critic.recurrent_cfg, intrinsic=cfg.ppo.use_intrinsic_loss)

    if dev.type == 'cuda':
        def gpu_sync_fn():
            torch.cuda.synchronize()
    else:
        def gpu_sync_fn():
            pass

    replay_buffer = None #NStepReplay(rollout_mgr, buffer_size, 'cuda')

    _update_loop(
        update_iter_fn=_update_iter,
        gpu_sync_fn=gpu_sync_fn,
        user_cb=update_cb,
        cfg=cfg,
        num_agents=num_agents,
        sim=sim,
        rollout_mgr=rollout_mgr,
        learning_state=learning_state,
        start_update_idx=start_update_idx,
        replay_buffer=replay_buffer,
    )

    return actor_critic

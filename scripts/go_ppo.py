import madrona_puzzle_bench
from madrona_puzzle_bench import SimFlags, RewardMode

from madrona_puzzle_bench_learn import (
    train, profile, TrainConfig, PPOConfig, SimInterface,
)

from policy import make_policy, setup_obs

import torch
import wandb
import argparse
import math
from pathlib import Path
import warnings
warnings.filterwarnings("error")
import numpy as np
import time

torch.manual_seed(0)

arg_parser = argparse.ArgumentParser()
# General args
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--ckpt-dir', type=str, required=True)
arg_parser.add_argument('--run-name', type=str, required=True)
arg_parser.add_argument('--profile-report', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')

# World gen args
arg_parser.add_argument('--use-fixed-world', action='store_true')
arg_parser.add_argument('--start-in-discovered-rooms', action='store_true')
arg_parser.add_argument('--reward-mode', type=str, required=True)
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--width-frac', type=float, default=1.0)
arg_parser.add_argument('--num-levels', type=int, default=1)

# Learning args
arg_parser.add_argument('--num-updates', type=int, required=True)
arg_parser.add_argument('--steps-per-update', type=int, default=40)
arg_parser.add_argument('--num-bptt-chunks', type=int, default=8)
arg_parser.add_argument('--lr', type=float, default=1e-4)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--entropy-loss-coef', type=float, default=0.01)
arg_parser.add_argument('--value-loss-coef', type=float, default=0.5)
arg_parser.add_argument('--clip-value-loss', action='store_true')
arg_parser.add_argument('--no-value-norm', action='store_true')
arg_parser.add_argument('--no-advantage-norm', action='store_true')
arg_parser.add_argument('--no-advantages', action='store_true')
arg_parser.add_argument('--value-normalizer-decay', type=float, default=0.999)
arg_parser.add_argument('--restore', type=int)
arg_parser.add_argument('--use-complex-level', action='store_true')
arg_parser.add_argument('--use-intrinsic-loss', action='store_true')

# Architecture args
arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')

# Go-Explore args
arg_parser.add_argument('--binning', type=str, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--num-bins', type=int, required=True)
arg_parser.add_argument('--num-checkpoints', type=int, default=1)
arg_parser.add_argument('--new-frac', type=float, default=0.5)
arg_parser.add_argument('--bin-reward-type', type=str, default="none")
arg_parser.add_argument('--bin-reward-boost', type=float, default=0.01)
arg_parser.add_argument('--uncertainty-metric', type=str, default="none")
arg_parser.add_argument('--buffer-strategy', type=str, default="fifo")
arg_parser.add_argument('--sampling-strategy', type=str, default="uniform")

# Binning diagnostic args
arg_parser.add_argument('--bin-diagnostic', action='store_true')
arg_parser.add_argument('--seeds-per-checkpoint', type=int, default=16)

args = arg_parser.parse_args()

normalize_values = not args.no_value_norm
normalize_advantages = not args.no_advantage_norm

sim_flags = SimFlags.Default
print(sim_flags)
if args.use_fixed_world:
    sim_flags |= SimFlags.UseFixedWorld
if args.start_in_discovered_rooms:
    sim_flags |= SimFlags.StartInDiscoveredRooms
if args.use_complex_level:
    sim_flags |= SimFlags.UseComplexLevel
print(sim_flags)

reward_mode = getattr(RewardMode, args.reward_mode)

if torch.cuda.is_available():
    dev = torch.device(f'cuda:{args.gpu_id}')
else:
    dev = torch.device('cpu')

ckpt_dir = Path(args.ckpt_dir)

ckpt_dir.mkdir(exist_ok=True, parents=True)

from torch.distributions.categorical import Categorical

class Box:
    def __init__(self, low, high, shape, dtype=np.float32):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype=dtype

class DiscreteActionDistributions:
    def __init__(self, actions_num_buckets, logits = None):
        self.actions_num_buckets = actions_num_buckets

        self.dists = []
        cur_bucket_offset = 0

        for num_buckets in self.actions_num_buckets:
            self.dists.append(Categorical(logits = logits[
                :, cur_bucket_offset:cur_bucket_offset + num_buckets],
                validate_args=False))
            cur_bucket_offset += num_buckets

    def best(self, out):
        actions = [dist.probs.argmax(dim=-1) for dist in self.dists]
        torch.stack(actions, dim=1, out=out)

    def sample(self):
        actions = [dist.sample() for dist in self.dists]
        return torch.stack(actions, dim=1)

    def action_stats(self, actions):
        log_probs = []
        entropies = []
        for i, dist in enumerate(self.dists):
            log_probs.append(dist.log_prob(actions[:, i]))
            entropies.append(dist.entropy())

        return torch.stack(log_probs, dim=1), torch.stack(entropies, dim=1)

    def probs(self):
        return [dist.probs for dist in self.dists]

class GoExplore:
    def __init__(self, num_worlds, device):
        self.worlds = madrona_puzzle_bench.SimManager(
            exec_mode = madrona_puzzle_bench.madrona.ExecMode.CUDA if args.gpu_sim else madrona_puzzle_bench.madrona.ExecMode.CPU,
            gpu_id = args.gpu_id,
            num_worlds = args.num_worlds,
            rand_seed = 5,
            sim_flags = (int)(sim_flags),
            reward_mode = reward_mode,
            episode_len = 200,
            levels_per_episode = args.num_levels,
            button_width = 1.3 * args.width_frac,
            door_width = 20.0 / 3.,
            reward_per_dist = 0.05,
            slack_reward = -0.005,
        )
        self.worlds.init()

        # Do the warm up
        use_warm_up = True
        if use_warm_up:
            steps_so_far = 0
            warm_up = 32
            while steps_so_far < 200:
                for i in range(warm_up - 1):
                    self.worlds.step()
                resets = self.worlds.reset_tensor().to_torch().view(-1)
                total_envs = resets.shape[0]
                reset_min = (steps_so_far / 200)*(1. - args.new_frac) + args.new_frac # Added the last bit to compensate for resets
                reset_max = ((steps_so_far + warm_up) / 200)*(1. - args.new_frac) + args.new_frac # Added the last bit to compensate for resets
                resets[(int)(reset_min * total_envs):(int)(reset_max * total_envs)] = 1
                print("Steps so far", steps_so_far)
                print("Max length", 200)
                print("Resetting", (int)(reset_min * total_envs), (int)(reset_max * total_envs))
                self.worlds.step()
                steps_so_far += warm_up

        self.num_worlds = num_worlds
        self.num_agents = 1
        self.curr_returns = torch.zeros(num_worlds, device = device) # For tracking cumulative return of each state/bin
        #print("Curr returns shape", self.curr_returns.shape)
        self.binning = args.binning
        self.num_bins = args.num_bins # We can change this later
        self.num_checkpoints = args.num_checkpoints
        self.device = device
        self.checkpoint_score = torch.zeros(self.num_bins, self.num_checkpoints, device=device)
        self.bin_count = torch.zeros(self.num_bins, device=device).int()
        self.max_return = 0
        self.max_progress = -1000

        self.obs, num_obs_features = setup_obs(self.worlds)
        self.policy = make_policy(num_obs_features, args.num_channels, args.separate_value, intrinsic=args.use_intrinsic_loss)
        self.actions = self.worlds.action_tensor().to_torch()
        self.dones = self.worlds.done_tensor().to_torch()
        self.rewards = self.worlds.reward_tensor().to_torch()
        self.checkpoints = self.worlds.checkpoint_tensor().to_torch()
        self.checkpoint_resets = self.worlds.checkpoint_reset_tensor().to_torch()
        self.resets = self.worlds.reset_tensor().to_torch()
        self.bin_checkpoints = torch.zeros((self.num_bins, self.num_checkpoints, self.checkpoints.shape[-1]), device=device, dtype=torch.uint8)
        self.bin_steps = torch.zeros((self.num_bins,), device=device).int() + 200
        self.start_bin_steps = torch.zeros((self.num_bins,), device=device).int() + 200
        self.world_steps = torch.zeros(self.num_worlds, device=device).int() + 200
        self.bin_reward_boost = args.bin_reward_boost
        self.reward_type = args.bin_reward_type

        # Add tracking for different types of uncertainties for resets
        self.bin_uncertainty = torch.zeros((self.num_bins, self.num_checkpoints), device=device).float() # Can have RND, TD error, (pseudo-counts? etc.)

        # Start bin
        start_bins = self.map_states_to_bins(self.obs)[0,:]
        self.start_bin_steps[start_bins] = 0

        self.actions_num_buckets = [4, 8, 5, 2]
        self.action_space = Box(-float('inf'),float('inf'),(sum(self.actions_num_buckets),))

        # Callback
        self.mean_fps = 0
        self.ckpt_dir = ckpt_dir
        self.profile_report = False

    # Corrected approach to get the first element of each group without using a for loop
    def get_first_elements_unsorted_groups(self, states, groups):
        # Sort groups and states based on groups
        sorted_groups, indices = groups.sort()
        sorted_states = states[indices]

        # Find the unique groups and the first occurrence of each group
        unique_groups, first_occurrences = torch.unique(sorted_groups, return_inverse=True)
        # Mask to identify first occurrences in the sorted array
        first_occurrence_mask = torch.zeros_like(sorted_groups, dtype=torch.bool).scatter_(0, first_occurrences, 1)

        return unique_groups, sorted_states[first_occurrence_mask]

    def generate_random_actions(self):
        action_dist = DiscreteActionDistributions(self.actions_num_buckets, logits=torch.ones(self.num_worlds, sum(self.actions_num_buckets), device=self.device))
        return action_dist.sample()

    # Step 1: Select state from archive
    # Uses: self.archive
    # Output: states
    def select_state(self, update_id):
        #print("About to select state")
        # First select from visited bins with go-explore weighting function
        valid_bins = torch.nonzero(self.bin_count > 0).flatten()
        if args.sampling_strategy == "uniform" or self.num_bins == 1:
            weights = 1./(torch.sqrt(self.bin_count[valid_bins])*0 + 1)
        elif args.sampling_strategy == "count":
            weights = 1./(torch.sqrt(self.bin_count[valid_bins]) + 1)
        elif args.sampling_strategy == "uncertainty":
            weights = (self.bin_uncertainty[valid_bins].sum(dim=-1) + 1)/(torch.clamp(self.bin_count[valid_bins], 0, self.num_checkpoints) + 1) # Doing mean by sum/count
        # Sample bins
        desired_samples = int(self.num_worlds*args.new_frac)#*((10001 - update_id) / 10000))
        sampled_bins = valid_bins[torch.multinomial(weights, num_samples=desired_samples, replacement=True).type(torch.int)]
        # Sample states from bins: either sample first occurrence in each bin (what's in the paper), or something better...
        # Need the last checkpoint for each bin
        if self.num_bins == 1:
            chosen_checkpoint = torch.randint(torch.clamp(self.bin_count[0] - 1, 0, self.num_checkpoints).item(), size=(desired_samples,), device=dev).type(torch.int)
        else:
            chosen_checkpoint = torch.randint(self.num_checkpoints, size=(desired_samples,), device=dev).type(torch.int) % self.bin_count[sampled_bins]

        #print("Bin count", self.bin_count[sampled_bins], self.bin_count[sampled_bins].shape)
        self.curr_returns[:desired_samples] = self.checkpoint_score[[sampled_bins, chosen_checkpoint]]
        #print("Checkpoints", self.bin_checkpoints[[sampled_bins, chosen_checkpoint]])
        return self.bin_checkpoints[[sampled_bins, chosen_checkpoint]]

    # Step 2: Go to state
    # Input: states, worlds
    # Logic: For each state, set world to state
    # Output: None
    def go_to_state(self, states, leave_rest, update_id):
        # Run checkpoint-restoration here
        #print("Before go-to-state")
        #print(self.obs[5][:])
        desired_samples = int(self.num_worlds*args.new_frac) # *((10001 - update_id) / 10000)) # Only set state for some worlds
        self.checkpoint_resets[:, 0] = 1
        self.checkpoints[:desired_samples] = states
        # Reset everything
        '''
        if not leave_rest:
            print("Resetting non-checkpoint worlds")
            goExplore.resets[desired_samples:, 0] = 1
            goExplore.checkpoint_resets[desired_samples:, 0] = 0
        '''
        self.worlds.step()
        self.obs[9][:desired_samples, 0] = 40 + torch.randint(0, 160, size=(desired_samples,), dtype=torch.int, device=dev) # Maybe set this to random 40 to 200? # 200
        #print("After go-to-state")
        #print(self.obs[5][:])
        return None

    # Step 3: Explore from state
    def explore_from_state(self):
        for i in range(self.num_exploration_steps):
            # Select actions either at random or by sampling from a trained policy
            self.actions[:] = self.generate_random_actions()
            self.worlds.step()
            # Map states to bins
            new_bins = self.map_states_to_bins(self.obs)
            # Update archive
            #print(self.curr_returns.shape)
            #print(self.rewards.view(self.num_worlds,self.num_agents).sum(dim=1).shape)
            self.curr_returns += self.rewards.view(self.num_worlds,self.num_agents).sum(dim=1).to(self.curr_returns.device)
            self.max_return = max(self.max_return, torch.max(self.curr_returns).item())
            #print(self.dones.shape)
            #print(self.dones)
            self.curr_returns *= (1 - 0.5*self.dones.view(self.num_worlds,self.num_agents).sum(dim=1)) # Account for dones, this is working!
            #print("Max return", torch.max(self.curr_returns), self.worlds.obs[torch.argmax(self.curr_returns)])
            self.update_archive(new_bins, self.curr_returns)
        return None

    def apply_binning_function(self, states):
        if self.binning == "none":
            return states
        elif self.binning == "random":
            return torch.randint(0, self.num_bins, size=(1, self.num_worlds,), device=self.device)
        elif self.binning == "y_pos":
            # Bin according to the y position of each agent
            # Determine granularity from num_bins
            granularity = torch.sqrt(torch.tensor(self.num_bins)).int().item()
            increment = 1.11/granularity
            #print("States shape", states[0].shape)
            self_obs = states[0].view(-1, self.num_worlds, 10)
            y_0 = torch.clamp((self_obs[..., 1] + 20)/40, 0, 1.1) // increment # Granularity of 0.01 on the y
            y_out = (y_0).int()
            #print("Max agent 0 progress", self_obs[:, 0, 3].max())
            #print("Max agent 1 progress", self_obs[:, 1, 3].max())
            # Also incorporate the scene id
            self_obs = states[2].view(-1, self.num_worlds)
            #print(states[2].shape)
            #print("Shapes", self_obs.shape, y_out.shape)
            return (y_out + granularity*self_obs).int()
        elif self.binning == "y_pos_door":
            # Bin according to the y position of each agent
            # Determine granularity from num_bins
            granularity = torch.sqrt(torch.tensor(self.num_bins)).int().item()
            increment = 1.11/granularity
            #print("States shape", states[0].shape)
            self_obs = states[0].view(-1, self.num_worlds, 10)
            y_0 = torch.clamp((self_obs[..., 1] + 20)/40, 0, 1.1) // increment # Granularity of 0.01 on the y
            y_out = (y_0).int()
            #print("Max agent 0 progress", self_obs[:, 0, 3].max())
            #print("Max agent 1 progress", self_obs[:, 1, 3].max())
            # Also incorporate the scene id
            level_obs = states[2].view(-1, self.num_worlds)
            #print(states[2].shape)
            # We get door open/button press from attr_1
            door_obs = states[6][...,0].max(dim=-1)[0].view(-1, self.num_worlds)
            #print("Shapes", self_obs.shape, y_out.shape)
            return (y_out + granularity*door_obs + granularity*2*level_obs).int()
        elif self.binning == "y_z_pos_door":
            # Bin according to the y position of each agent
            # Determine granularity from num_bins
            granularity = torch.sqrt(torch.tensor(self.num_bins)).int().item()
            increment = 1.11/granularity
            #print("States shape", states[0].shape)
            self_obs = states[0].view(-1, self.num_worlds, 10)
            y_0 = torch.clamp((self_obs[..., 1] + 20)/40, 0, 1.1) // increment # Granularity of 0.01 on the y
            y_out = (y_0).int()
            z_out = self_obs[...,2].int()
            #print("Max agent 0 progress", self_obs[:, 0, 3].max())
            #print("Max agent 1 progress", self_obs[:, 1, 3].max())
            # Also incorporate the scene id
            level_obs = states[2].view(-1, self.num_worlds)
            #print(states[2].shape)
            # We get door open/button press from attr_1
            door_obs = states[6][...,0].max(dim=-1)[0].view(-1, self.num_worlds)
            #print("Shapes", self_obs.shape, y_out.shape)
            return (y_out + granularity*z_out + granularity*5*door_obs + granularity*5*2*level_obs).int()
        elif self.binning == "x_y":
            # Bin according to the y position of each agent
            # Determine granularity from num_bins
            granularity = torch.sqrt(torch.tensor(self.num_bins)).int().item()
            increment = 1.11/granularity
            #print("States shape", states[0].shape)
            self_obs = states[0].view(-1, self.num_worlds, 10)
            y_0 = torch.clamp((self_obs[..., 1] + 20)/40, 0, 1.1) // increment # Granularity of 0.01 on the y
            y_out = (y_0).int()
            x_0 = torch.clamp((self_obs[..., 0] + 10)/20, 0, 1.1) // increment # Granularity of 0.01 on the x
            #print(self_obs[..., 0].min(), self_obs[..., 0].max())
            #print("Max agent 0 progress", self_obs[:, 0, 3].max())
            #print("Max agent 1 progress", self_obs[:, 1, 3].max())
            # Also incorporate the scene id
            #level_obs = states[2].view(-1, self.num_worlds)
            #print(states[2].shape)
            # We get door open/button press from attr_1
            #door_obs = states[6][...,0].max(dim=-1)[0].view(-1, self.num_worlds)
            #print("Shapes", self_obs.shape, y_out.shape)
            return (y_out + granularity*x_0).int()
        elif self.binning == "y_pos_door_block":
            # Bin according to the y position of each agent
            granularity = torch.sqrt(torch.tensor(self.num_bins) / 40).int().item()
            increment = 1.11/granularity
            self_obs = states[0].view(args.steps_per_update, self.num_worlds, self.num_agents, -1)
            y_0 = torch.clamp(self_obs[..., 0, 3], 0, 1.1) // increment # Granularity of 0.01 on the y
            y_1 = torch.clamp(self_obs[..., 1, 3], 0, 1.1) // increment # Granularity of 0.01 on the y
            #print("Max y progress", self_obs[:, 0, 3].max())
            # Now check if the door is open
            door_obs = states[3].view(args.steps_per_update, self.num_worlds, self.num_agents, -1)
            door_status = door_obs[..., 0, 2] + 2*door_obs[..., 1, 2]
            # Also bin block_pos since we want the blocks on the doors
            #print(states[2].shape)
            # Maybe for now average distance of the blocks from each agent
            block_obs = states[2].view(args.steps_per_update, self.num_worlds, self.num_agents, -1, 3)
            block_val = (block_obs[..., 2].mean(dim=2).sum(dim=2)*8).int() % 10
            #print("Block val", block_val.mean())
            #print(door_status)
            return (block_val*(granularity*granularity*4) + door_status*(granularity*granularity) + (y_0 + granularity*y_1)).int()
        else:
            raise NotImplementedError

    # Step 4: Map encountered states to bins
    def map_states_to_bins(self, states):
        # Apply binning function to define bin for new states
        bins = self.apply_binning_function(states)
        if torch.any(bins > self.num_bins):
            # throw error
            raise ValueError("Bin value too large")
        # Now return the binning of all states
        return bins

    def update_bin_steps(self, bins, prev_bins):
        #print(self.bin_steps[bins].shape, self.bin_steps[prev_bins].shape)
        #self.bin_steps[bins] = torch.minimum(self.bin_steps[bins], self.bin_steps[prev_bins] + 1)
        if self.num_bins > 1:
            for i in range(1, args.steps_per_update):
                self.bin_steps[prev_bins[-i]] = torch.minimum(self.bin_steps[prev_bins[-i]], self.bin_steps[bins[-i]] + 1)
                self.start_bin_steps[bins[i - 1]] = torch.minimum(self.start_bin_steps[bins[i - 1]], self.start_bin_steps[prev_bins[i - 1]] + 1)

    # Step 5: Update archive
    def update_archive(self, bins, scores, ppo_stats):
        # For each unique bin, update count in archive and update best score
        # Reshape ppo_stats to be per-world
        ppo_stats.intrinsic_reward_array = ppo_stats.intrinsic_reward_array.view(self.num_worlds, self.num_agents).mean(dim=1) if args.use_intrinsic_loss else None
        #print("Value loss array shape", ppo_stats.value_loss_array.shape)
        ppo_stats.value_loss_array = ppo_stats.value_loss_array.view(self.num_worlds, self.num_agents).mean(dim=1)
        # At most can increase bin count by 1 in a single step...
        desired_samples = int(self.num_worlds*args.new_frac) # Only store checkpoints from "fresh" worlds
        bins = bins[desired_samples:]
        if self.num_bins == 1:
            # Assumes num_worlds < num_checkpoints
            chosen_checkpoints = (self.bin_count[bins] + torch.arange(0, bins.shape[0], device=dev)) % self.num_checkpoints# Spread out the stored checkpoints 
            self.bin_count += bins.shape[0]
        else:
            new_bin_counts = (torch.bincount(bins, minlength=self.num_bins) > 0).int()
            # Set the checkpoint for each bin to the latest
            chosen_checkpoints = self.bin_count[bins] % self.num_checkpoints
            self.bin_count += new_bin_counts
        #print(chosen_checkpoints)
        #print(bins)
        stacked_indices = torch.stack((bins, chosen_checkpoints), dim=1)
        # https://stackoverflow.com/questions/72001505/how-to-get-unique-elements-and-their-firstly-appeared-indices-of-a-pytorch-tenso
        unique, idx, counts = torch.unique(stacked_indices, dim=0, sorted=True, return_inverse=True, return_counts=True) 
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=dev), cum_sum[:-1]))
        first_indices = ind_sorted[cum_sum]
        # Set up a sample filter if we keep the most uncertain or highest score, not fifo
        if args.buffer_strategy == "uncertainty":
            # Check if current sample is more uncertain than the one in the bin
            if args.uncertainty_metric == "rnd":
                curr_uncertainty = ppo_stats.intrinsic_reward_array[desired_samples:][first_indices].to(dev)
            elif args.uncertainty_metric == "td_error":
                curr_uncertainty = ppo_stats.value_loss_array[desired_samples:][first_indices].to(dev)
            else:
                raise NotImplementedError
            bin_uncertainty = self.bin_uncertainty[[unique[:,0], unique[:,1]]]
            sample_filter = curr_uncertainty > bin_uncertainty
        elif args.buffer_strategy == "score":
            # Check if current sample has a higher score than the one in the bin
            sample_filter = scores[desired_samples:][first_indices] >= self.checkpoint_score[[unique[:,0], unique[:,1]]] # TODO: get scores to work properly
        elif args.buffer_strategy == "fifo":
            # Just take the first samples
            sample_filter = torch.ones_like(first_indices, dtype=torch.bool, device=dev)
        else:
            raise NotImplementedError
        # Apply sample filter to unique and first_indices
        unique = unique[sample_filter]
        first_indices = first_indices[sample_filter]
        #print(unique, first_indices)
        self.bin_checkpoints[[unique[:,0], unique[:,1]]] = self.checkpoints[desired_samples:][first_indices.to(self.checkpoints.device)].to(dev)
        #self.bin_score[bins] = torch.maximum(self.bin_score[bins], scores)
        self.checkpoint_score[[unique[:,0], unique[:,1]]] = scores[desired_samples:][first_indices] # Track checkpoint scores, we might as well
        # Also update bin uncertainties
        if args.uncertainty_metric == "rnd":
            # Update RND
            self.bin_uncertainty[[unique[:,0], unique[:,1]]] = ppo_stats.intrinsic_reward_array[desired_samples:][first_indices].to(dev)
        elif args.uncertainty_metric == "td_error":
            # Update TD error
            self.bin_uncertainty[[unique[:,0], unique[:,1]]] = ppo_stats.value_loss_array[desired_samples:][first_indices].to(dev)
        # Otherwise we're doing counts, and that requires nothing from us here
        return None

    # Compute best score from archive
    def compute_best_score(self):
        return torch.max(self.state_score)
    
    # Learning callback
    def __call__(self, update_idx, update_time, update_results, learning_state):
        update_id = update_idx + 1
        fps = args.num_worlds * args.steps_per_update / update_time
        self.mean_fps += (fps - self.mean_fps) / update_id

        skip_log = False
        if update_id != 1 and update_id % 10 != 0:
            skip_log = True

        ppo = update_results.ppo_stats

        if not skip_log:
            # Only log stuff from the worlds where we're not setting state
            desired_samples = int(self.num_worlds*args.new_frac)*self.num_agents#*((10001 - update_id) / 10000))*self.num_agents
            print("Desired samples", desired_samples)
            with torch.no_grad():
                print(update_results.rewards.shape)
                reward_mean = update_results.rewards[:,desired_samples:].mean().cpu().item()
                reward_min = update_results.rewards[:,desired_samples:].min().cpu().item()
                reward_max = update_results.rewards[:,desired_samples:].max().cpu().item()

                done_count = (update_results.dones[:,desired_samples:] == 1.0).sum()
                return_mean, return_min, return_max, success_frac = 0, 0, 0, 0
                #print("Update results shape", update_results.returns.shape)
                if done_count > 0:
                    #print("Update results shape", update_results.returns[update_results.dones == 1.0].shape)
                    return_mean = update_results.returns[:,desired_samples:][update_results.dones[:,desired_samples:] == 1.0].mean().cpu().item()
                    return_min = update_results.returns[:,desired_samples:][update_results.dones[:,desired_samples:] == 1.0].min().cpu().item()
                    return_max = update_results.returns[:,desired_samples:][update_results.dones[:,desired_samples:] == 1.0].max().cpu().item()
                    #success_frac = update_results.obs[0][update_results.dones == 1.0, ..., 3].mean().cpu().item()
                    #print((update_results.obs[0][...,3] > 1.00).shape)
                    #print((update_results.obs[0][...,3] > 1.00)[:,:,desired_samples:].shape)
                    success_filter = (update_results.dones[:,desired_samples:] == 1.0)[...,0]
                    #print(success_filter.shape)
                    success_frac = (update_results.rewards[:,desired_samples:] >= 1.00).reshape(-1, success_filter.shape[-1])[success_filter].float().mean().cpu().item() # Reward of 1 for room, 10 for whole set of rooms
                    # Break this down for each level type, which is obs[2]
                    level_type_success_fracs = []
                    for i in range(8):
                        level_type_success_fracs.append((update_results.rewards[:,desired_samples:][(update_results.obs[2][0][:,desired_samples:] == i)*(update_results.dones[:,desired_samples:] == 1.0)] >= 1.00).float().mean().cpu().item())
                else:
                    success_frac = 0
                    level_type_success_fracs = [0, 0, 0, 0, 0, 0]
                # Also compute level type success without filtering by desired_samples
                success_filter_all = (update_results.dones == 1.0)[...,0]
                success_frac_all = (update_results.rewards >= 1.00).reshape(-1, success_filter_all.shape[-1])[success_filter_all].float().mean().cpu().item() # Reward of 1 for room, 10 for whole set of rooms
                level_type_success_fracs_all = []
                if success_filter_all.sum() > 0:
                    for i in range(8):
                        level_type_success_fracs_all.append((update_results.rewards[(update_results.obs[2][0] == i)*(update_results.dones == 1.0)] >= 1.00).float().mean().cpu().item())
                else:
                    level_type_success_fracs_all = [0, 0, 0, 0, 0, 0, 0, 0]

                # compute visits to second and third room
                #print("Update results shape", update_results.obs[0].shape, update_results.obs[3].shape)

                value_mean = update_results.values[:,desired_samples:].mean().cpu().item()
                value_min = update_results.values[:,desired_samples:].min().cpu().item()
                value_max = update_results.values[:,desired_samples:].max().cpu().item()

                advantage_mean = update_results.advantages[:,desired_samples:].mean().cpu().item()
                advantage_min = update_results.advantages[:,desired_samples:].min().cpu().item()
                advantage_max = update_results.advantages[:,desired_samples:].max().cpu().item()

                vnorm_mu = 0
                vnorm_sigma = 0
                if normalize_values:
                    vnorm_mu = learning_state.value_normalizer.mu.cpu().item()
                    vnorm_sigma = learning_state.value_normalizer.sigma.cpu().item()

            print(f"\nUpdate: {update_id}")
            print(f"    Loss: {ppo.loss: .3e}, A: {ppo.action_loss: .3e}, V: {ppo.value_loss: .3e}, E: {ppo.entropy_loss: .3e}")
            print()
            print(f"    Rewards          => Avg: {reward_mean: .3e}, Min: {reward_min: .3e}, Max: {reward_max: .3e}")
            print(f"    Values           => Avg: {value_mean: .3e}, Min: {value_min: .3e}, Max: {value_max: .3e}")
            print(f"    Advantages       => Avg: {advantage_mean: .3e}, Min: {advantage_min: .3e}, Max: {advantage_max: .3e}")
            print(f"    Returns          => Avg: {return_mean}, max: {return_max}")
            print(f"    Value Normalizer => Mean: {vnorm_mu: .3e}, σ: {vnorm_sigma :.3e}")

            # Log average steps to end from known bins
            avg_steps = self.bin_steps[self.bin_count > 0].float().mean()

            # Add all this to wandb
            wandb.log({
                "update_id": update_id,
                "loss": ppo.loss,
                "action_loss": ppo.action_loss,
                "value_loss": ppo.value_loss,
                "entropy_loss": ppo.entropy_loss,
                "reward_mean": reward_mean, 
                "reward_max": reward_max,
                "returns_mean": return_mean,
                "returns_max": return_max,
                "done_count": done_count,
                "vnorm_mu": vnorm_mu,
                "steps_to_end": avg_steps,
                "max_progress": self.max_progress,
                "success_frac": success_frac,
                # Add logging of success frac for each level type
                "success_frac_0": level_type_success_fracs[0],
                "success_frac_1": level_type_success_fracs[1],
                "success_frac_2": level_type_success_fracs[2],
                "success_frac_3": level_type_success_fracs[3],
                "success_frac_4": level_type_success_fracs[4],
                "success_frac_5": level_type_success_fracs[5],
                "success_frac_6": level_type_success_fracs[6],
                "success_frac_7": level_type_success_fracs[7],
                # Also log unfiltered versions of success_fracs
                "success_frac_all": success_frac_all,
                "success_frac_0_all": level_type_success_fracs_all[0],
                "success_frac_1_all": level_type_success_fracs_all[1],
                "success_frac_2_all": level_type_success_fracs_all[2],
                "success_frac_3_all": level_type_success_fracs_all[3],
                "success_frac_4_all": level_type_success_fracs_all[4],
                "success_frac_5_all": level_type_success_fracs_all[5],
                "success_frac_6_all": level_type_success_fracs_all[6],
                "success_frac_7_all": level_type_success_fracs_all[7],
                }
            )
            if args.use_intrinsic_loss:
                wandb.log({
                    "update_id": update_id,
                    "intrinsic_loss": ppo.intrinsic_loss,
                    "value_loss_intrinsic": ppo.value_loss_intrinsic,
                })

            if self.profile_report:
                print()
                print(f"    FPS: {fps:.0f}, Update Time: {update_time:.2f}, Avg FPS: {self.mean_fps:.0f}")
                print(f"    PyTorch Memory Usage: {torch.cuda.memory_reserved() / 1024 / 1024 / 1024:.3f}GB (Reserved), {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.3f}GB (Current)")
                profile.report()

            if update_id % 100 == 0:
                learning_state.save(update_idx, self.ckpt_dir / f"{update_id}.pth")

        #if update_id % 5 != 0:
        #    return

        # Now do the go-explore stuff
        goExplore.max_progress = max(goExplore.max_progress, update_results.obs[0][...,1].max())
        '''
        if goExplore.max_progress > 30:
            # Print where this occurs
            print(torch.where(update_results.obs[0][...,1] > 30))
            # Something is wrong, dump checkpoints
            print("Dumping checkpoints")
            with open("max_progress_dump" + str(update_id), 'wb') as f:
                goExplore.checkpoints.cpu().numpy().tofile(f)
            raise ValueError("Max progress too high")
        '''
        if goExplore.max_progress > 1.01: # Do this based off the exit observation
            exit_bins = goExplore.map_states_to_bins(goExplore.obs)[0,:][(goExplore.obs[0][...,3] > 1.01).view(goExplore.num_worlds, goExplore.num_agents).all(dim=1)]
            # Set exit path length to 0 for exit bins
            goExplore.bin_steps[exit_bins] = 0
            print("Exit bins", torch.unique(exit_bins).shape)
            #writer.add_scalar("charts/exit_path_length", goExplore.bin_steps[exit_bins].float().mean(), global_step)

        # Update archive from rollout
        #new_bins = self.map_states_to_bins(self.obs)
        all_bins = self.map_states_to_bins(update_results.obs) # num_timesteps * num_worlds
        self.update_bin_steps(all_bins[1:], all_bins[:-1])
        new_bins = all_bins[-1]
        # Update archive
        #print(self.curr_returns.shape)
        #print(self.rewards.view(self.num_worlds,self.num_agents).sum(dim=1).shape)
        self.curr_returns += self.rewards.view(self.num_worlds,self.num_agents).sum(dim=1).to(self.curr_returns.device)
        self.max_return = max(self.max_return, torch.max(self.curr_returns).item())
        #print(self.dones.shape)
        #print(self.dones)
        self.curr_returns *= (1 - 0.5*self.dones.view(self.num_worlds,self.num_agents).sum(dim=1)).to(self.curr_returns.device) # Account for dones, this is working!
        #print("Max return", torch.max(self.curr_returns), self.worlds.obs[torch.argmax(self.curr_returns)])
        self.update_archive(new_bins, self.curr_returns, ppo)
        # Set new state, go to state
        if args.new_frac > 0.002:
            states = self.select_state(update_id)
            self.go_to_state(states, leave_rest = (update_id % 5 != 0), update_id = update_id)

# Maybe we can just write Go-Explore as a callback

# Before training: initialize archive
# During training:
#   1. Run rollout + PPO
#   2. Update archive from rollout
#   3. Run diagnostic if desired
#   4. Select next state for next rollout

# Now run the train loop from the other script
# Create GoExplore object from args
goExplore = GoExplore(args.num_worlds, dev)

if args.restore:
    restore_ckpt = ckpt_dir / f"{args.restore}.pth"
else:
    restore_ckpt = None

run = wandb.init(
    # Set the project where this run will be logged
    project="escape-room-ppo-go",
    # Track hyperparameters and run metadata
    config=args
)

train(
    dev,
    SimInterface(
            step = lambda: goExplore.worlds.step(),
            obs = goExplore.obs,
            actions = goExplore.actions,
            dones = goExplore.dones,
            rewards = goExplore.rewards,
            resets = goExplore.resets,
            checkpoints = goExplore.checkpoints,
            checkpoint_resets = goExplore.checkpoint_resets
    ),
    TrainConfig(
        run_name = args.run_name,
        num_updates = args.num_updates,
        steps_per_update = args.steps_per_update,
        num_bptt_chunks = args.num_bptt_chunks,
        lr = args.lr,
        gamma = args.gamma,
        gae_lambda = 0.95,
        ppo = PPOConfig(
            num_mini_batches=1,
            clip_coef=0.2,
            value_loss_coef=args.value_loss_coef,
            use_intrinsic_loss=args.use_intrinsic_loss, 
            value_loss_intrinsic_coef=0.5, # TODO: Need to set
            intrinsic_loss_coef=0.5, # TODO: Need to set
            entropy_coef=args.entropy_loss_coef,
            max_grad_norm=0.5,
            num_epochs=2,
            clip_value_loss=args.clip_value_loss,
            no_advantages=args.no_advantages,
        ),
        value_normalizer_decay = args.value_normalizer_decay,
        mixed_precision = args.fp16,
        normalize_advantages = normalize_advantages,
        normalize_values = normalize_values,
    ),
    goExplore.policy,
    goExplore,
    restore_ckpt
)

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
import os

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
arg_parser.add_argument('--no-level-obs', action='store_true')
arg_parser.add_argument('--seed', type=int, default=5)

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
arg_parser.add_argument('--no-onehot', action='store_true')

# Architecture args
arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--entity-network', action='store_true')

# Go-Explore args
arg_parser.add_argument('--binning', type=str, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--num-bins', type=int, required=True)
arg_parser.add_argument('--num-checkpoints', type=int, default=1)
arg_parser.add_argument('--new-frac', type=float, default=0.0001)
arg_parser.add_argument('--bin-reward-type', type=str, default="none")
arg_parser.add_argument('--bin-reward-boost', type=float, default=0.01)
arg_parser.add_argument('--uncertainty-metric', type=str, default="none")
arg_parser.add_argument('--buffer-strategy', type=str, default="fifo")
arg_parser.add_argument('--sampling-strategy', type=str, default="uniform")
arg_parser.add_argument('--make-graph', action='store_true')
arg_parser.add_argument('--importance-test', action='store_true')
arg_parser.add_argument('--importance-weights', action='store_true')
arg_parser.add_argument('--intelligent-sample', action='store_true')
arg_parser.add_argument('--intelligent-weights', nargs='+', type=int, default=[1, 1, 1, 1, 1, 1])
arg_parser.add_argument('--td-weights', action='store_true')
arg_parser.add_argument('--choose-worlds', action='store_true')
arg_parser.add_argument('--world-success-frac', type=float, default=0.5)
arg_parser.add_argument('--success-weights', action='store_true')
arg_parser.add_argument('--success-rate-temp', type=float, default=1.0)

# Binning diagnostic args
arg_parser.add_argument('--bin-diagnostic', action='store_true')
arg_parser.add_argument('--seeds-per-checkpoint', type=int, default=16)

args = arg_parser.parse_args()

#args.num_updates = args.num_updates // 2 # Temporary change for script, will need to roll back

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
            rand_seed = args.seed,
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

        if args.entity_network:
            self.obs, num_obs_features, num_entity_features = setup_obs(self.worlds, args.no_level_obs, use_onehot=False, separate_entity=True)
            self.policy = make_policy(num_obs_features, num_entity_features, args.num_channels, args.separate_value, intrinsic=args.use_intrinsic_loss)
        else:
            self.obs, num_obs_features = setup_obs(self.worlds, args.no_level_obs)
            self.policy = make_policy(num_obs_features, None, args.num_channels, args.separate_value, intrinsic=args.use_intrinsic_loss)
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
        self.new_frac = args.new_frac
        self.importance_test = args.importance_test
        self.importance_weights = args.importance_weights
        self.intelligent_sample = args.intelligent_sample
        self.intelligent_weights = args.intelligent_weights
        self.td_weights = args.td_weights
        self.running_td_error = torch.tensor(args.intelligent_weights, device = self.checkpoints.device).float() # Just for initialization, we'll do EMA on this
        self.running_success_rate = torch.tensor(args.intelligent_weights, device = self.checkpoints.device).float() * 0 # Just for initialization, we'll do EMA on this
        self.success_weights = args.success_weights
        self.success_rate_temp = args.success_rate_temp

        # Temporarily store checkpoints for 1) current worlds, 2) failed worlds, 3) successful worlds
        self.choose_worlds = args.choose_worlds
        self.world_success_frac = args.world_success_frac
        self.current_worlds = self.checkpoints.clone().detach()
        self.failure_worlds = None
        self.success_worlds = None

        # Add tracking for different types of uncertainties for resets
        self.bin_uncertainty = torch.zeros((self.num_bins, self.num_checkpoints), device=device).float() # Can have RND, TD error, (pseudo-counts? etc.)

        # Start bin
        start_bins = self.map_states_to_bins(self.obs)[0,:]
        self.start_bin_steps[start_bins] = 0

        self.actions_num_buckets = [4, 8, 5, 2]
        self.action_space = Box(-float('inf'),float('inf'),(sum(self.actions_num_buckets),))

        # If we want to fully discretize the MDP then we also extract a graph of the transitions
        if args.make_graph:
            self.transition_graph = torch.zeros((self.num_bins, self.num_bins), device=device).int()

        # Callback
        self.mean_fps = 0
        self.ckpt_dir = ckpt_dir
        self.profile_report = False

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
            return torch.randint(0, self.num_bins, size=(states[0].shape[1], self.num_worlds,), device=self.device)
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
            granularity = (torch.tensor(self.num_bins) / 2).int().item()
            increment = 1.11/granularity
            #print("States shape", states[0].shape)
            self_obs = states[0].view(-1, self.num_worlds, 10)
            y_0 = torch.clamp((self_obs[..., 1] + 20)/40, 0, 1.1) // increment # Granularity of 0.01 on the y
            y_out = (y_0).int()
            # We get door open/button press from attr_1
            door_obs = states[6][...,0].max(dim=-1)[0].view(-1, self.num_worlds)
            #print("Shapes", self_obs.shape, y_out.shape)
            return (y_out + granularity*door_obs).int()
        elif self.binning == "y_pos_door_level":
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
        elif self.binning == "y_pos_door_entities":
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
            # Entity obs
            entity_obs = states[4].view(-1, self.num_worlds, 9, states[4].shape[-1])
            # Normalize this
            entity_sum_dist = torch.abs(entity_obs[..., 0]).min(dim=2)[0] + torch.abs(entity_obs[..., 1]).min(dim=2)[0]
            entity_obs_norm = (entity_sum_dist - entity_sum_dist.min())/(entity_sum_dist.max() - entity_sum_dist.min())
            block_val = (entity_obs_norm * 10).int() % 10
            #print("Shapes", self_obs.shape, y_out.shape)
            return (y_out + granularity*door_obs + granularity*2*level_obs + granularity*2*8*block_val).int()
        elif self.binning == "block_button":
            # Let's make a task-specific binning function
            granularity = (torch.tensor(1000) / 5).int().item()#(torch.tensor(self.num_bins) / 5).int().item()
            increment = 1.01/granularity
            #print(stage_obs)
            # Now we want to have distance to goal in here
            entity_obs = states[4].view(-1, self.num_worlds, 9, states[4].shape[-1])#[..., 0].min(dim=2)[0]
            entity_type = states[5].view(-1, self.num_worlds, 9)
            block_id = 3
            button_id = 6
            # Creating a meshgrid for B and C dimensions
            B, C = entity_obs.shape[:2]
            b_grid, c_grid = torch.meshgrid(torch.arange(B), torch.arange(C), indexing='ij')
            block_dist = entity_obs[b_grid,c_grid,(entity_type == block_id).float().argmax(dim=-1)][...,0:2].norm(dim=-1).view(-1, self.num_worlds)
            button_dist = entity_obs[entity_type == button_id][...,0:2].norm(dim=-1).view(-1, self.num_worlds)
            block_button_dist = (entity_obs[entity_type == button_id][...,0:2].view(-1, self.num_worlds, 2) - entity_obs[b_grid,c_grid,(entity_type == block_id).float().argmax(dim=-1)][...,0:2]).norm(dim=-1).view(-1, self.num_worlds)
            exit_dist = states[3][...,0].view(-1, self.num_worlds)
            # Three stages: 1) go to block, 2) drag block to button, 3) go to open door
            door_obs = block_button_dist < 1.0 # Seems reasonable ...
            grab_obs = states[1][...,0].view(-1, self.num_worlds) #.max(dim=-1)[0].view(-1, self.num_worlds)
            #print("Door obs", door_obs, grab_obs)
            # 0 if neither, 1 if grab, 2 if door open (and 3 if door and grab)
            stage_obs = (grab_obs + 2*door_obs).int()#.clamp(0, 2)
            # Now generate a binning from stage and stage_dist
            dist_obs = (stage_obs == 0)*block_dist + (stage_obs == 1)*block_button_dist + (stage_obs == 2)*exit_dist + (stage_obs == 3)*exit_dist
            # Need more granularity for stage_obs == 3
            # Now bin by stage_obs and dist_obs, assume dist_obs < 25
            dist_quantized = (1.0 - torch.clamp(dist_obs / 25, 0., 1.)) // increment
            print("Max stage obs", stage_obs.max())
            bins = (stage_obs*granularity + dist_quantized).int()
            print(torch.unique(bins))
            return bins
        elif self.binning == "block_button_new":
            # Let's make a task-specific binning function
            granularity = (torch.tensor(1000) / 5).int().item()#(torch.tensor(self.num_bins) / 5).int().item()
            increment = 1.01/granularity
            #print(stage_obs)
            # Now we want to have distance to goal in here
            entity_obs = states[4].view(-1, self.num_worlds, 9, states[4].shape[-1])#[..., 0].min(dim=2)[0]
            entity_type = states[5].view(-1, self.num_worlds, 9)
            block_id = 3
            button_id = 6
            # Creating a meshgrid for B and C dimensions
            B, C = entity_obs.shape[:2]
            b_grid, c_grid = torch.meshgrid(torch.arange(B), torch.arange(C), indexing='ij')
            block_dist = entity_obs[b_grid,c_grid,(entity_type == block_id).float().argmax(dim=-1)][...,0:2].norm(dim=-1).view(-1, self.num_worlds)
            button_dist = entity_obs[entity_type == button_id][...,0:2].norm(dim=-1).view(-1, self.num_worlds)
            block_button_dist = (entity_obs[entity_type == button_id][...,0:2].view(-1, self.num_worlds, 2) - entity_obs[b_grid,c_grid,(entity_type == block_id).float().argmax(dim=-1)][...,0:2]).norm(dim=-1).view(-1, self.num_worlds)
            exit_dist = states[3][...,0].view(-1, self.num_worlds)
            # Three stages: 1) go to block, 2) drag block to button, 3) go to open door
            door_obs = block_button_dist < 1.0 # Seems reasonable ...
            grab_obs = states[1][...,0].view(-1, self.num_worlds) #.max(dim=-1)[0].view(-1, self.num_worlds)
            door_open_obs = states[6][...,0].max(dim=-1)[0].view(-1, self.num_worlds)
            agent_block_button_obs = (block_button_dist < 2.0) * (button_dist < 5.0)
            # 0 if none, 1 if grab_obs, 2 if agent_block_button_obs, 3 if door_open_obs and not agent_block_button_obs
            stage_obs = ((grab_obs * (~agent_block_button_obs) * (1 - door_open_obs)) + 2*agent_block_button_obs + 2*(door_open_obs * (~agent_block_button_obs))).int()
            #print("Door obs", door_obs, grab_obs)
            print("Max stage obs", stage_obs.max(), "Min stage obs", stage_obs.min())
            print("Without rounding", ((grab_obs * (~agent_block_button_obs) * (1 - door_open_obs)) + 2*agent_block_button_obs + 2*(door_open_obs * (~agent_block_button_obs))).min())
            bins = (stage_obs*granularity + 5).int()
            #print(torch.unique(bins))
            return bins
        elif self.binning == "block_button_keystages":
            # Let's make a task-specific binning function
            granularity = (torch.tensor(1000) / 5).int().item()#(torch.tensor(self.num_bins) / 5).int().item()
            increment = 1.01/granularity
            #print(stage_obs)
            # Now we want to have distance to goal in here
            entity_obs = states[4].view(-1, self.num_worlds, 9, states[4].shape[-1])#[..., 0].min(dim=2)[0]
            entity_type = states[5].view(-1, self.num_worlds, 9)
            block_id = 3
            button_id = 6
            # Creating a meshgrid for B and C dimensions
            B, C = entity_obs.shape[:2]
            b_grid, c_grid = torch.meshgrid(torch.arange(B), torch.arange(C), indexing='ij')
            block_dist = entity_obs[b_grid,c_grid,(entity_type == block_id).float().argmax(dim=-1)][...,0:2].norm(dim=-1).view(-1, self.num_worlds)
            button_dist = entity_obs[entity_type == button_id][...,0:2].norm(dim=-1).view(-1, self.num_worlds)
            block_button_dist = (entity_obs[entity_type == button_id][...,0:2].view(-1, self.num_worlds, 2) - entity_obs[b_grid,c_grid,(entity_type == block_id).float().argmax(dim=-1)][...,0:2]).norm(dim=-1).view(-1, self.num_worlds)
            exit_dist = states[3][...,0].view(-1, self.num_worlds)
            # Three stages: 1) go to block, 2) drag block to button, 3) go to open door
            door_obs = states[6][...,0].max(dim=-1)[0].view(-1, self.num_worlds)
            grab_obs = states[1][...,0].view(-1, self.num_worlds) #.max(dim=-1)[0].view(-1, self.num_worlds)
            # Key stage 1: Near block, not grabbing
            stage_1 = (block_dist < 3.0)*(grab_obs == 0)
            # Key stage 2: Grabbing block, near button, door not open
            stage_2 = (block_button_dist < 3.0)*(grab_obs == 1)*(door_obs == 0)
            # Key stage 3: Grabbing block, door open
            stage_3 = (door_obs == 1)*(grab_obs == 1)
            return (stage_1*granularity + stage_2*granularity*2 + stage_3*granularity*3).int()
        elif self.binning == "lava":
            granularity = 200
            # Let's make a task-specific binning function
            self_obs = states[0].view(-1, self.num_worlds, 10)
            y_pos = self_obs[..., 1]
            before_lava = (y_pos < -15).int()
            after_lava = (y_pos >= 13).int()
            # Now if in-lava, let's see whether there's room behind, forward, or to the side...
            # Forward is 0, backwards is -pi/pi. 
            agent_theta = (self_obs[..., -1] * -1) + (np.pi / 30)
            agent_bin = agent_theta / (np.pi / 15)
            agent_bin[agent_bin < 0] += 30
            agent_bin = agent_bin.int()
            # Now access lidar depth forward
            lidar_depth = states[7].view(-1, self.num_worlds, 30)
            lidar_hit_type = states[8].view(-1, self.num_worlds, 30)
            print("Lidar depth", lidar_depth.shape)
            print("Agent bin", agent_bin.shape)
            agent_bin = agent_bin.view(-1, self.num_worlds, 1).type(torch.cuda.LongTensor).to(lidar_depth.device)
            forward_depth = torch.gather(lidar_depth, 2, agent_bin)[...,0]
            #sideways_depth = torch.maximum(lidar_depth[(agent_bin + 7) % 30], lidar_depth[(agent_bin + 23) % 30])
            sideways_depth = torch.maximum(torch.gather(lidar_depth, 2, (agent_bin + 7) % 30), torch.gather(lidar_depth, 2, (agent_bin + 23) % 30))[...,0]
            # Check if forward is wall or lava
            #forward_wall = (lidar_hit_type[agent_bin] == 9).int()
            forward_wall = (torch.gather(lidar_hit_type, 2, agent_bin)[...,0] == 9).int()
            # stage_obs
            # 1: Before lava
            # 2: After lava
            # 3: In lava, forward room, no forward wall
            # 4: In lava, sideways room, no forward wall
            # 5: In lava, forward room, forward wall
            print("Before lava", before_lava.shape)
            print("After lava", after_lava.shape)
            print("Forward depth", forward_depth.shape)
            print("Forward wall", forward_wall.shape)
            print("Sideways depth", sideways_depth.shape)
            stage_obs = 1*before_lava + 2*after_lava + 3*(before_lava == 0)*(after_lava == 0)*(forward_depth > 3.0)*(forward_wall == 0) + 4*(before_lava == 0)*(after_lava == 0)*(forward_depth > 3.0)*(forward_wall == 1) + 5*(before_lava == 0)*(after_lava == 0)*(sideways_depth > 3.0)*(forward_depth <= 3.0)
            # How do we define success? 
            # If there's forward room, we want to make forward progress without death
            # If there's sideways room, we again want to make forward progress without death
            # If before lava, want to enter lava region ... again can be measured by forward progress without death 
            bins = (stage_obs*granularity + 5).int()
            return bins
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
    def map_states_to_bins(self, states, max_bin = True):
        # Apply binning function to define bin for new states
        bins = self.apply_binning_function(states)
        # Apply mod to bring below num_bins
        if max_bin:
            bins = bins % self.num_bins
            if torch.any(bins > self.num_bins):
                # throw error
                raise ValueError("Bin value too large")
        # Now return the binning of all states
        return bins

    def update_bin_steps(self, bins, prev_bins, dones):
        #print(self.bin_steps[bins].shape, self.bin_steps[prev_bins].shape)
        #self.bin_steps[bins] = torch.minimum(self.bin_steps[bins], self.bin_steps[prev_bins] + 1)
        if self.num_bins > 1:
            for i in range(1, args.steps_per_update):
                self.bin_steps[prev_bins[-i]] = torch.minimum(self.bin_steps[prev_bins[-i]], self.bin_steps[bins[-i]] + 1)
                self.start_bin_steps[bins[i - 1]] = torch.minimum(self.start_bin_steps[bins[i - 1]], self.start_bin_steps[prev_bins[i - 1]] + 1)
        
        if args.make_graph:
            # Update transition graph
            self.transition_graph[prev_bins[dones[...,0] == 0], bins[dones[...,0] == 0]] += 1

    # Step 5: Update archive
    def update_archive(self, bins, scores, ppo_stats):
        # For each unique bin, update count in archive and update best score
        # Reshape ppo_stats to be per-world
        ppo_stats.intrinsic_reward_array = ppo_stats.intrinsic_reward_array.view(self.num_worlds, self.num_agents).mean(dim=1) if args.use_intrinsic_loss else None
        #print("Value loss array shape", ppo_stats.value_loss_array.shape)
        if args.uncertainty_metric == "td_error":
            ppo_stats.value_loss_array = ppo_stats.value_loss_array.view(self.num_worlds, self.num_agents).mean(dim=1) # THIS IS BROKEN WITH MULTIPLE MINIBATCHES
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
                    success_filter = (update_results.dones[:,desired_samples:] == 1.0)[...,0].sum(dim=0) > 0

                    # Use success_filter to move these checkpoints into success and failure worlds
                    solved_filter = (update_results.rewards[:,desired_samples:] >= 1.00).reshape(-1, success_filter.shape[-1]).sum(dim=0) > 0

                    # Now copy worlds from current_worlds to success_worlds and failure_worlds
                    max_size = 100000
                    if success_filter.sum() > 0 and self.choose_worlds:
                        if self.success_worlds is None:
                            self.success_worlds = self.current_worlds[desired_samples:][success_filter*solved_filter]
                        else:
                            self.success_worlds = torch.cat((self.success_worlds, self.current_worlds[desired_samples:][success_filter*solved_filter]), dim=0)[:max_size]
                        if self.failure_worlds is None:
                            self.failure_worlds = self.current_worlds[desired_samples:][success_filter*(~solved_filter)]
                        else:
                            self.failure_worlds = torch.cat((self.failure_worlds, self.current_worlds[desired_samples:][success_filter*(~solved_filter)]), dim=0)[:max_size]

                        # Now copy over the checkpoints for the completed worlds to current_worlds
                        self.current_worlds[desired_samples:][success_filter] = self.checkpoints[desired_samples:][success_filter].clone().detach()

                    # Now set all other worlds from just success worlds or just failure worlds
                    if self.choose_worlds:
                        self.checkpoint_resets[:, 0] = 1
                        success_samples = (int)(self.world_success_frac*desired_samples)
                        # Adjust success_samples to reflect the sizes of the buffers--lower if no successes available, and increase if no failures availble
                        if self.success_worlds is not None and self.success_worlds.shape[0] < success_samples:
                            success_samples = self.success_worlds.shape[0]
                        elif self.failure_worlds is not None and self.failure_worlds.shape[0] < (desired_samples - success_samples):
                            success_samples = desired_samples - self.failure_worlds.shape[0]

                        if self.world_success_frac > 0.0 and self.success_worlds is not None and self.success_worlds.shape[0] > 0:
                            # Take the first desired_samples from success worlds, or repeat if fewer to get desired_samples
                            if self.success_worlds.shape[0] < success_samples:
                                self.checkpoints[:success_samples] = self.success_worlds.repeat((success_samples // self.success_worlds.shape[0]) + 1, 1)[:success_samples]
                            else:
                                # Use randperm to select worlds
                                self.checkpoints[:success_samples] = self.success_worlds[torch.randperm(self.success_worlds.shape[0])[:success_samples]]
                        if self.world_success_frac < 1.0 and self.failure_worlds is not None and self.failure_worlds.shape[0] > 0:
                            # Take the first desired_samples from failure worlds, or repeat if fewer to get desired_samples
                            failure_samples = desired_samples - success_samples
                            # Remove selected worlds from failure worlds
                            if self.failure_worlds.shape[0] < failure_samples:
                                self.checkpoints[success_samples:desired_samples] = self.failure_worlds.repeat((failure_samples // self.failure_worlds.shape[0]) + 1, 1)[:failure_samples]
                                #self.failure_worlds = None
                            else:
                                self.checkpoints[success_samples:desired_samples] = self.failure_worlds[torch.randperm(self.failure_worlds.shape[0])[:failure_samples]]
                                # Shift failure_worlds to delete selected worlds
                                #self.failure_worlds = self.failure_worlds[failure_samples:]
                        self.worlds.step()

                    success_filter = (update_results.dones[:,desired_samples:] == 1.0)[...,0]

                    #print(success_filter.shape)
                    success_frac = (update_results.rewards[:,desired_samples:] >= 8.00).reshape(-1, success_filter.shape[-1])[success_filter].float().mean().cpu().item() # Reward of 1 for room, 10 for whole set of rooms
                    # Break this down for each level type, which is obs[2]
                    level_type_success_fracs = []
                    for i in range(8):
                        level_type_success_fracs.append((update_results.rewards[:,desired_samples:][(update_results.obs[2][0][:,desired_samples:] == i)*(update_results.dones[:,desired_samples:] == 1.0)] >= 8.00).float().mean().cpu().item())
                else:
                    success_frac = 0
                    level_type_success_fracs = [0, 0, 0, 0, 0, 0, 0, 0]
                # Also compute level type success without filtering by desired_samples
                success_filter_all = (update_results.dones == 1.0)[...,0]
                success_frac_all = (update_results.rewards >= 8.00).reshape(-1, success_filter_all.shape[-1])[success_filter_all].float().mean().cpu().item() # Reward of 1 for room, 10 for whole set of rooms
                level_type_success_fracs_all = []
                if success_filter_all.sum() > 0:
                    for i in range(8):
                        level_type_success_fracs_all.append((update_results.rewards[(update_results.obs[2][0] == i)*(update_results.dones == 1.0)] >= 8.00).float().mean().cpu().item())
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

                # Now let's compute a bunch of metrics per stage-level bin
                all_bins = self.map_states_to_bins(update_results.obs, max_bin=False)
                # Let's do bin-counts here for now, TODO have to reverse this
                new_bin_counts = torch.bincount(all_bins.flatten(), minlength=self.num_bins)# > 0).int()
                #self.bin_count += torch.clone(new_bin_counts)
                new_bin_counts = new_bin_counts.float()

                # Map to super-bins
                all_bins = all_bins // 200
                # Assume this is in range 0-3, let's track TD error and intrinsic loss for each of these
                all_td_error = [0, 0, 0, 0, 0, 0]
                all_td_var = [0, 0, 0, 0, 0, 0]
                all_value_mean = [0, 0, 0, 0, 0, 0]
                all_value_var = [0, 0, 0, 0, 0, 0]
                all_intrinsic_loss = [0, 0, 0, 0, 0, 0]
                all_entropy_loss = [0, 0, 0, 0, 0, 0]
                all_bin_variance = [0, 0, 0, 0, 0, 0]
                all_success_rate = [0, 0, 0, 0, 0, 0]
                all_bin_counts = [0, 0, 0, 0, 0, 0]
                for i in range(6):
                    bin_filter = (all_bins == i)
                    all_td_error[i] = ppo.value_loss_array[bin_filter].mean().cpu().item()
                    all_value_mean[i] = update_results.values[bin_filter].mean().cpu().item()
                    if bin_filter.sum() > 1:
                        all_td_var[i] = ppo.value_loss_array[bin_filter].var().cpu().item()
                        all_value_var[i] = update_results.values[bin_filter].var().cpu().item()
                    if args.use_intrinsic_loss:
                        all_intrinsic_loss[i] = ppo.intrinsic_reward_array[bin_filter].mean().cpu().item()
                    # Update running estimates of TD error if not nan
                    if not math.isnan(all_td_error[i]):
                        self.running_td_error[i] = 0.9*self.running_td_error[i] + 0.1*all_td_error[i]

                    # Also set up estimates of entropy loss per bin, and bin count variance per bin
                    all_entropy_loss[i] = ppo.entropy_loss_array[bin_filter].mean().cpu().item()

                    # Compute starting_bins
                    starting_bin_filter = (all_bins[0] == i)
                    print("Starting bin filter", starting_bin_filter.shape)
                    print("All bins", all_bins.shape)
                    print("starting bin sum", starting_bin_filter.sum())

                    all_bin_counts[i] = starting_bin_filter.sum()
                    forward_progress_sum = update_results.rewards[:, starting_bin_filter].sum(axis = 0)
                    print("Forward progress sum", torch.mean(forward_progress_sum), forward_progress_sum)
                    # Reward per dist = 0.05, entire lava area is 30 units, let's make it 0.05*10 = 1.0. We won't get it if we die lol
                    death_filter = (update_results.rewards[:, starting_bin_filter] < -0.5).max(dim = 0)[0]
                    # Reset filter
                    reset_filter = ((update_results.dones[:,starting_bin_filter] == 1.0)[...,0]).sum(dim=0) > 0
                    forward_progress_sum[reset_filter] = forward_progress_sum[reset_filter] * (forward_progress_sum[reset_filter] > 8.0) # Keep only the successes
                    all_success_rate[i] = ((forward_progress_sum >= 0.5) * (1 - death_filter.int())).float().mean().cpu().item()

                    ''' # For simple block-button and obstructed block-button
                    all_bin_counts[i] = starting_bin_filter.sum().cpu().item()
                    if i < 2:
                        success_bin_filter = (all_bins[:, starting_bin_filter] == i + 1).max(dim = 0)
                        all_success_rate[i] = success_bin_filter[0].float().mean().cpu().item()
                        print("Success rate", all_success_rate[i])
                    else:
                        # Check if there is a reward == 10 somewhere in there
                        success_bin_filter = (update_results.rewards[:, starting_bin_filter] >= 8.00).max(dim = 0)
                        all_success_rate[i] = success_bin_filter[0].float().mean().cpu().item()
                    '''
                    
                    # Running success rate
                    if not math.isnan(all_success_rate[i]):
                        self.running_success_rate[i] = 0.9*self.running_success_rate[i] + 0.1*all_success_rate[i]

                    #if bin_filter.sum() > 1 and args.binning != "random":
                    #    all_bin_variance[i] = torch.std(new_bin_counts[200*i:200*(i+1)]).cpu().item() / torch.mean(new_bin_counts[200*i:200*(i+1)]).cpu().item()
                print("All td error, running", all_td_error, self.running_td_error)
                print("All success rate, running", all_success_rate, self.running_success_rate)
                # Turn off intelligent sampling if success_frac_all > 0.75
                #if success_frac_all > 0.8:
                #    self.intelligent_sample = False
                #    self.new_frac = 0.0

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
                "steps_from_start": self.start_bin_steps[self.bin_count > 0].float().mean(),
                "max_progress": self.max_progress,
                "success_frac": success_frac,
                # Log go-explore stats
                "discovered_bins": (self.bin_count > 0).sum(),
                "max_bin": torch.nonzero(self.bin_count > 0).max() if (self.bin_count > 0).sum() > 0 else 0,
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
                # Now log the TD error for each stage-level bin
                "td_error_0": all_td_error[0],
                "td_error_1": all_td_error[1],
                "td_error_2": all_td_error[2],
                "td_error_3": all_td_error[3],
                "td_error_4": all_td_error[4],
                "td_error_5": all_td_error[5],
                # Now log the TD variance for each stage-level bin
                "td_var_0": all_td_var[0],
                "td_var_1": all_td_var[1],
                "td_var_2": all_td_var[2],
                "td_var_3": all_td_var[3],
                "td_var_4": all_td_var[4],
                "td_var_5": all_td_var[5],
                # Now log the value mean for each stage-level bin
                "value_mean_0": all_value_mean[0],
                "value_mean_1": all_value_mean[1],
                "value_mean_2": all_value_mean[2],
                "value_mean_3": all_value_mean[3],
                "value_mean_4": all_value_mean[4],
                "value_mean_5": all_value_mean[5],
                # Now log the value variance for each stage-level bin
                "value_var_0": all_value_var[0],
                "value_var_1": all_value_var[1],
                "value_var_2": all_value_var[2],
                "value_var_3": all_value_var[3],
                "value_var_4": all_value_var[4],
                "value_var_5": all_value_var[5],
                # Now log the entropy loss for each stage-level bin
                "entropy_loss_0": all_entropy_loss[0],
                "entropy_loss_1": all_entropy_loss[1],
                "entropy_loss_2": all_entropy_loss[2],
                "entropy_loss_3": all_entropy_loss[3],
                "entropy_loss_4": all_entropy_loss[4],
                "entropy_loss_5": all_entropy_loss[5],
                # Report the bin variance
                "bin_variance_0": all_bin_variance[0],
                "bin_variance_1": all_bin_variance[1],
                "bin_variance_2": all_bin_variance[2],
                "bin_variance_3": all_bin_variance[3],
                "bin_variance_4": all_bin_variance[4],
                "bin_variance_5": all_bin_variance[5],
                # Report success rate for each stage-level bin
                "success_rate_0": all_success_rate[0],
                "success_rate_1": all_success_rate[1],
                "success_rate_2": all_success_rate[2],
                "success_rate_3": all_success_rate[3],
                "success_rate_4": all_success_rate[4],
                "success_rate_5": all_success_rate[5],
                # Also bin counts
                "bin_count_0": all_bin_counts[0],
                "bin_count_1": all_bin_counts[1],
                "bin_count_2": all_bin_counts[2],
                "bin_count_3": all_bin_counts[3],
                "bin_count_4": all_bin_counts[4],
                "bin_count_5": all_bin_counts[5],
                }
            )
            if args.use_intrinsic_loss:
                wandb.log({
                    "update_id": update_id,
                    "intrinsic_loss": ppo.intrinsic_loss,
                    "value_loss_intrinsic": ppo.value_loss_intrinsic,
                    # Now log the intrinsic loss for each stage-level bin
                    "intrinsic_loss_0": all_intrinsic_loss[0],
                    "intrinsic_loss_1": all_intrinsic_loss[1],
                    "intrinsic_loss_2": all_intrinsic_loss[2],
                    "intrinsic_loss_3": all_intrinsic_loss[3],
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
        self.max_progress = max(self.max_progress, update_results.obs[0][...,1].max())
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
        if False:
            # Save one checkpoint from every bin
            print("Dumping checkpoints")
            nonzero_bins = torch.nonzero(self.bin_count > 0).flatten()
            # Get path from ckpt arg
            save_dir = "ge_ckpts_" + str(self.ckpt_dir).split("/")[-1]
            # Make save_dir if it doesn't exist
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            with open(save_dir + "/checkpoints" + str(update_id) + "_" + str(nonzero_bins.shape[0]), 'wb') as f:
                #b_grid, c_grid = torch.meshgrid(nonzero_bins, torch.zeros_like(nonzero_bins), indexing='ij')
                self.bin_checkpoints[nonzero_bins, 0].reshape(nonzero_bins.shape[0], -1).cpu().numpy().tofile(f)
                #self.checkpoints.cpu().numpy().tofile(f)
            # Also write the bin graph to file
            if args.make_graph:
                np.savez(save_dir + "/graph" + str(update_id) + ".npy", self.transition_graph.cpu().numpy(), self.bin_steps.cpu().numpy())
        # VISHNU: TEMPORARY CHANGE FOR DEBUGGING
        '''
        exit_bins = self.map_states_to_bins(update_results.obs)[(update_results.rewards >= 9.00)[...,0]]
        if exit_bins.shape[0] > 0: # Do this based off the exit observation
            # Set exit path length to 0 for exit bins
            self.bin_steps[exit_bins] = 0
            print("Exit bins", torch.unique(exit_bins).shape)

        # Update archive from rollout
        #new_bins = self.map_states_to_bins(self.obs)
        all_bins = self.map_states_to_bins(update_results.obs) # num_timesteps * num_worlds
        self.update_bin_steps(all_bins[1:], all_bins[:-1], update_results.dones[:-1])
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
        '''
        ''' // Moved to train.py
        if args.new_frac > 0.002:
            desired_samples = int(self.num_worlds*args.new_frac)
            self.checkpoint_resets[:, 0] = 1
            self.checkpoints[:desired_samples] = self.checkpoints[desired_samples:].repeat((self.num_worlds // (self.num_worlds - desired_samples)) - 1, 1)
            self.worlds.step()
        '''

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
    project="puzzle-bench-stages",
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
            num_mini_batches=1, # VISHNU: WAS 1
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

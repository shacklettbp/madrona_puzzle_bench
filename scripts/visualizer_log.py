import madrona_puzzle_bench

from madrona_puzzle_bench_learn import LearningState, SimInterface

from policy import make_policy, setup_obs

import torch
import numpy as np
import argparse
import math
from pathlib import Path
import warnings
warnings.filterwarnings("error")

from madrona_puzzle_bench_learn.rollouts import RolloutManager, Rollouts
from madrona_puzzle_bench_learn.amp import AMPState
from madrona_puzzle_bench_learn.moving_avg import EMANormalizer
from torch import optim
import datetime
import json

torch.manual_seed(0)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--ckpt-path', type=str, required=True)
arg_parser.add_argument('--run-name', type=str, required=True)
arg_parser.add_argument('--action-dump-path', type=str)

arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)

arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')

arg_parser.add_argument('--gpu-sim', action='store_true')

args = arg_parser.parse_args()

sim = madrona_puzzle_bench.SimManager(
    exec_mode = madrona_puzzle_bench.madrona.ExecMode.CUDA if args.gpu_sim else madrona_puzzle_bench.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    auto_reset = True,
)

obs, num_obs_features = setup_obs(sim)
policy = make_policy(num_obs_features, args.num_channels, args.separate_value)

dev = torch.device(f'cuda:{args.gpu_id}')
optimizer = optim.Adam(policy.parameters(), lr=0.1)
amp = AMPState(dev, True)
value_normalizer = EMANormalizer(0.0,
                                    disable=False)
value_normalizer = value_normalizer.to(dev)

learning_state = LearningState(
    policy = policy,
    optimizer = optimizer,
    scheduler = None,
    value_normalizer = value_normalizer,
    amp = amp,
)

start_update_idx = learning_state.load(args.ckpt_path)

actions = sim.action_tensor().to_torch()
dones = sim.done_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()

# Flatten N, A, ... tensors to N * A, ...
actions = actions.view(-1, *actions.shape[2:])
dones  = dones.view(-1, *dones.shape[2:])
rewards = rewards.view(-1, *rewards.shape[2:])

cur_rnn_states = []

for shape in policy.recurrent_cfg.shapes:
    cur_rnn_states.append(torch.zeros(
        *shape[0:2], actions.shape[0], shape[2], dtype=torch.float32, device=dev))

sim_int = SimInterface(
            step = lambda: sim.step(),
            obs = obs,
            actions = actions,
            dones = dones,
            rewards = rewards,
    )

rollout_mgr = RolloutManager(dev, sim_int, args.num_steps,
        1, amp, policy.recurrent_cfg)

with torch.no_grad():
    rollouts = rollout_mgr.collect(amp, sim_int, policy.to(dev), value_normalizer)

# Dump the features
now = datetime.datetime.now()
dir_path = "/data/rl/madrona_3d_example/scripts/"
# torch.save(rollouts, dir_path + str(now) + ".pt")

trajectories_data = []
    
values = rollouts.values.cpu().reshape((args.num_steps, -1))
rewards = rollouts.rewards.cpu().reshape((args.num_steps, -1))
returns = rollouts.returns.cpu().reshape((args.num_steps, -1))
positions = rollouts.obs[0].cpu().reshape((args.num_steps, -1, rollouts.obs[0].shape[-1]))
door_opens = rollouts.obs[3].cpu()[:,:,:,2].reshape((args.num_steps, -1))

for i in range(args.num_worlds):
    trajectory = []
    for j in range(args.num_steps):
        x = positions[j, i, 2]
        y = positions[j, i, 3]
        value = values[j, i]
        reward = rewards[j, i]
        forward_return = returns[-1, i] - returns[j, i]
        door_open = door_opens[j, i]
        trajectory.append({"x": float(x), "y": float(y), "value": float(value), "reward": float(reward), "door_open": float(door_open), "return": float(forward_return)})
    trajectories_data.append(trajectory)

trajectories_json = json.dumps(trajectories_data)
folder_name = "/data/rl/madrona_3d_example/visualization/" + args.run_name + "/"
Path(folder_name).mkdir(parents=True, exist_ok=True)
with open(folder_name + args.ckpt_path.split('/')[-1].split('.')[0] + ".json", "w") as json_file:
    json_file.write(trajectories_json)
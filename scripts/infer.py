import torch
import madrona_puzzle_bench
from madrona_puzzle_bench import SimFlags, RewardMode
from madrona_puzzle_bench_learn import LearningState

from policy import make_policy, setup_obs

import numpy as np
import argparse
import math
from pathlib import Path
import warnings
warnings.filterwarnings("error")

torch.manual_seed(0)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--ckpt-path', type=str, required=True)
arg_parser.add_argument('--record-log', type=str)

arg_parser.add_argument('--use-fixed-world', action='store_true')

arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--no-level-obs', action='store_true')

arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')

arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--use-complex-level', action='store_true')


args = arg_parser.parse_args()

sim_flags = madrona_puzzle_bench.SimFlags.Default
if args.use_complex_level:
    sim_flags |= madrona_puzzle_bench.SimFlags.UseComplexLevel
print(sim_flags)

sim = madrona_puzzle_bench.SimManager(
    exec_mode = madrona_puzzle_bench.madrona.ExecMode.CUDA if args.gpu_sim else madrona_puzzle_bench.madrona.ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    rand_seed = 10,
    sim_flags = (int)(sim_flags),
    reward_mode = getattr(RewardMode, "Dense1"),
    episode_len = 200,
    levels_per_episode = 1,
    button_width = 1.3,
    door_width = 20.0 / 3.,
    reward_per_dist = 0.05,
    slack_reward = -0.005,
)
sim.init()

obs, num_obs_features = setup_obs(sim, args.no_level_obs)
policy = make_policy(num_obs_features, None, args.num_channels, args.separate_value)

weights = LearningState.load_policy_weights(args.ckpt_path)
policy.load_state_dict(weights)

policy = policy.to(torch.device(f"cuda:{args.gpu_id}")) if args.gpu_sim else policy.to(torch.device('cpu'))

actions = sim.action_tensor().to_torch()
dones = sim.done_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()

ckpts = sim.checkpoint_tensor().to_torch()

cur_rnn_states = []

for shape in policy.recurrent_cfg.shapes:
    cur_rnn_states.append(torch.zeros(
        *shape[0:2], actions.shape[0], shape[2], dtype=torch.float32, device=torch.device('cpu')))

if args.record_log:
    record_log = open(args.record_log, 'wb')
else:
    record_log = None

for i in range(args.num_steps):
    with torch.no_grad():
        action_dists, values, cur_rnn_states = policy(cur_rnn_states, *obs)
        #action_dists.best(actions)
        # Make placeholders for actions_out and log_probs_out
        if True:
            log_probs_out = torch.zeros_like(actions).float()
            action_dists.sample(actions, log_probs_out)
        else:
            action_dists.best(actions)

        probs = action_dists.probs()

    if record_log:
        ckpts.cpu().numpy().tofile(record_log)

    '''
    print()
    print("Self:", obs[0])
    print("Partners:", obs[1])
    print("Room Entities:", obs[2])
    print("Lidar:", obs[3])

    print("Move Amount Probs")
    print(" ", np.array_str(probs[0][0].cpu().numpy(), precision=2, suppress_small=True))
    print(" ", np.array_str(probs[0][1].cpu().numpy(), precision=2, suppress_small=True))

    print("Move Angle Probs")
    print(" ", np.array_str(probs[1][0].cpu().numpy(), precision=2, suppress_small=True))
    print(" ", np.array_str(probs[1][1].cpu().numpy(), precision=2, suppress_small=True))

    print("Rotate Probs")
    print(" ", np.array_str(probs[2][0].cpu().numpy(), precision=2, suppress_small=True))
    print(" ", np.array_str(probs[2][1].cpu().numpy(), precision=2, suppress_small=True))

    print("Grab Probs")
    print(" ", np.array_str(probs[3][0].cpu().numpy(), precision=2, suppress_small=True))
    print(" ", np.array_str(probs[3][1].cpu().numpy(), precision=2, suppress_small=True))

    print("Actions:\n", actions.cpu().numpy())
    print("Values:\n", values.cpu().numpy())
    '''
    sim.step()
    print("Rewards:\n", rewards)

if record_log:
    record_log.close()

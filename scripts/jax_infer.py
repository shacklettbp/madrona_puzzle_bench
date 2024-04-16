import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
import numpy as np

import argparse
from functools import partial

import madrona_puzzle_bench
from madrona_puzzle_bench import SimFlags, RewardMode
from madrona_puzzle_bench.madrona import ExecMode

import madrona_learn

from jax_policy import make_policy

madrona_learn.init(0.6)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, default=200)

arg_parser.add_argument('--ckpt-path', type=str, required=True)
arg_parser.add_argument('--single-policy', type=int, default=None)
arg_parser.add_argument('--record-log', type=str)

arg_parser.add_argument('--print-obs', action='store_true')
arg_parser.add_argument('--print-action-probs', action='store_true')
arg_parser.add_argument('--print-rewards', action='store_true')

arg_parser.add_argument('--fp16', action='store_true')
arg_parser.add_argument('--bf16', action='store_true')
arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--gpu-id', type=int, default=0)

args = arg_parser.parse_args()

dev = jax.devices()[0]

if args.fp16:
    dtype = jnp.float16
elif args.bf16:
    dtype = jnp.bfloat16
else:
    dtype = jnp.float32

policy = make_policy(dtype)

if args.single_policy != None:
    policy_states, num_policies = madrona_learn.eval_load_ckpt(
        policy, args.ckpt_path, single_policy = args.single_policy)
else:
    policy_states, num_policies = madrona_learn.eval_load_ckpt(
        policy, args.ckpt_path, train_only=True)

print(policy_states.reward_hyper_params)

sim = madrona_puzzle_bench.SimManager(
    exec_mode = ExecMode.CUDA if args.gpu_sim else ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    num_pbt_policies = 0,
    rand_seed = 10,
    sim_flags = SimFlags.Default,
    reward_mode = RewardMode.Dense1,
    episode_len = 200,
    levels_per_episode = 1,
    button_width = 1.3,
    door_width = 20.0 / 3.,
    reward_per_dist = 0.05,
    slack_reward = -0.005,
)

ckpt_tensor = sim.checkpoint_tensor()

team_size = 1
num_teams = 1

num_agents_per_world = team_size * num_teams

jax_gpu = jax.devices()[0].platform == 'gpu'

sim_dict = sim.jax(jax_gpu)
sim_init = sim_dict['init']
sim_step = sim_dict['step']

if args.record_log:
    record_log_file = open(args.record_log, 'wb')
else:
    record_log_file = None

step_idx = 0

def host_cb(obs, actions, action_probs, values, dones, rewards):
    global step_idx

    if args.print_obs:
        print(obs)

    #print(f"\nStep {step_idx}")

    if args.print_action_probs:
        for i in range(actions.shape[0]):
            if i % num_agents_per_world == 0:
                print(f"World {i // num_agents_per_world}")

            print(f" Agent {i % num_agents_per_world}:")
            print("  Action:", actions[..., i, :])

            print(f"  Move Amount Probs: {float(action_probs[0][i][0]):.2e} {float(action_probs[0][i][1]):.2e} {float(action_probs[0][i][2]):.2e}")
            print(f"  Turn Probs:        {float(action_probs[1][i][0]):.2e} {float(action_probs[1][i][1]):.2e} {float(action_probs[1][i][2]):.2e}")

    if args.print_rewards:
        print("Rewards:", rewards)

    np.array(ckpt_tensor.to_jax()).tofile(record_log_file)

    step_idx += 1

    return ()

def iter_cb(step_data):
    cb = partial(jax.experimental.io_callback, host_cb, ())

    cb(step_data['obs'],
       step_data['actions'],
       step_data['action_probs'],
       step_data['values'],
       step_data['dones'],
       step_data['rewards'])

cfg = madrona_learn.EvalConfig(
    num_worlds = args.num_worlds,
    team_size = team_size,
    num_teams = num_teams,
    num_eval_steps = args.num_steps,
    eval_competitive = False,
    policy_dtype = dtype,
)

episode_scores = policy_states.episode_score

print(episode_scores.mean)

episode_scores = madrona_learn.eval_policies(
    dev, cfg, sim_dict, policy, policy_states, iter_cb)

print(episode_scores.mean)

del sim

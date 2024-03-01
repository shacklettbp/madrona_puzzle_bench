import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn
from flax.metrics import tensorboard

import argparse
from functools import partial
from time import time
import os

import madrona_puzzle_bench
from madrona_puzzle_bench import SimFlags
from madrona_puzzle_bench.madrona import ExecMode

import madrona_learn
from madrona_learn import (
    TrainConfig, CustomMetricConfig, PPOConfig, PBTConfig, ParamRange,
)

from jax_policy import make_policy
from common import print_elos

madrona_learn.init(0.6)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--ckpt-dir', type=str, required=True)
arg_parser.add_argument('--tb-dir', type=str, required=True)
arg_parser.add_argument('--run-name', type=str, required=True)
arg_parser.add_argument('--restore', type=int)

arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-updates', type=int, required=True)
arg_parser.add_argument('--steps-per-update', type=int, default=40)
arg_parser.add_argument('--num-bptt-chunks', type=int, default=8)
arg_parser.add_argument('--num-minibatches', type=int, default=2)

arg_parser.add_argument('--lr', type=float, default=1e-4)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--entropy-loss-coef', type=float, default=0.01)
arg_parser.add_argument('--value-loss-coef', type=float, default=0.5)
arg_parser.add_argument('--clip-value-loss', action='store_true')

arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')
arg_parser.add_argument('--bf16', action='store_true')

arg_parser.add_argument('--pbt-ensemble-size', type=int, default=0)

arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--profile-port', type=int, default=None)

args = arg_parser.parse_args()

sim = madrona_puzzle_bench.SimManager(
    exec_mode = ExecMode.CUDA if args.gpu_sim else ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    auto_reset = True,
    num_pbt_policies = args.pbt_ensemble_size,
    rand_seed = 5,
    sim_flags = SimFlags.StaggerStarts | SimFlags.RandomFlipTeams,
)

jax_gpu = jax.devices()[0].platform == 'gpu'

sim_init, sim_step = sim.jax(jax_gpu)

tb_writer = tensorboard.SummaryWriter(os.path.join(args.tb_dir, args.run_name))

def metrics_cb(metrics, epoch, mb, train_state):
    return metrics

last_time = 0
last_update = 0

def host_cb(update_id, metrics, train_state_mgr):
    global last_time, last_update

    cur_time = time()
    update_diff = update_id - last_update

    print(f"Update: {update_id}")
    if last_time != 0:
        print("  FPS:", args.num_worlds * args.steps_per_update * update_diff / (cur_time - last_time))

    last_time = cur_time
    last_update = update_id

    metrics.pretty_print()
    vnorm_mu = train_state_mgr.train_states.value_normalizer_state['mu'][0][0]
    vnorm_sigma = train_state_mgr.train_states.value_normalizer_state['sigma'][0][0]
    print(f"    Value Normalizer => Mean: {vnorm_mu: .3e}, σ: {vnorm_sigma: .3e}")

    lrs = train_state_mgr.train_states.hyper_params.lr
    entropy_coefs = train_state_mgr.train_states.hyper_params.entropy_coef
    reward_hyper_params = train_state_mgr.policy_states.reward_hyper_params

    print(lrs)
    print(entropy_coefs)
    print(reward_hyper_params[..., 0])
    print(reward_hyper_params[..., 1])

    elos = train_state_mgr.policy_states.fitness_score[..., 0]
    print_elos(elos)

    print()

    metrics.tensorboard_log(tb_writer, update_id)

    for i in range(elos.shape[0]):
        tb_writer.scalar(f"p{i}/elo", elos[i], update_id)
        tb_writer.scalar(f"p{i}/team_spirit",
                         reward_hyper_params[i][0], update_id)
        tb_writer.scalar(f"p{i}/hit_reward_scale",
                         reward_hyper_params[i][1], update_id)

    num_train_policies = lrs.shape[0]
    for i in range(lrs.shape[0]):
        tb_writer.scalar(f"p{i}/lr", lrs[i], update_id)
        tb_writer.scalar(f"p{i}/entropy_coef", entropy_coefs[i], update_id)

    if update_id % 500 == 0:
        train_state_mgr.save(update_id,
            f"{args.ckpt_dir}/{args.run_name}/{update_id}")

    return ()

def iter_cb(update_idx, metrics, train_state_mgr):
    cb = partial(jax.experimental.io_callback, host_cb, ())
    noop = lambda *args: ()

    update_id = update_idx + 1

    should_callback = jnp.logical_or(update_id == 1, update_id % 10 == 0)

    lax.cond(should_callback, cb, noop,
             update_id, metrics, train_state_mgr)

    return should_callback

dev = jax.devices()[0]

if args.pbt_ensemble_size != 0:
    pbt_cfg = PBTConfig(
        num_teams = 1,
        team_size = 1,
        num_train_policies = args.pbt_ensemble_size,
        num_past_policies = 0,
        train_policy_cull_interval = 20,
        num_cull_policies = 1,
        self_play_portion = 1,
        lr_explore_range = ParamRange(
            min = 0.1,
            max = 10.0,
            log10_space = True,
        ),
        entropy_explore_range = ParamRange(
            min = 0.1,
            max = 10.0,
            log10_space = True,
        )
    )
else:
    pbt_cfg = None

if args.fp16:
    dtype = jnp.float16
elif args.bf16:
    dtype = jnp.bfloat16
else:
    dtype = jnp.float32

cfg = TrainConfig(
    num_worlds = args.num_worlds,
    num_agents_per_world = 1,
    num_updates = args.num_updates,
    steps_per_update = args.steps_per_update,
    num_bptt_chunks = args.num_bptt_chunks,
    lr = args.lr,
    gamma = args.gamma,
    gae_lambda = 0.95,
    algo = PPOConfig(
        num_mini_batches = args.num_minibatches,
        clip_coef = 0.2,
        value_loss_coef = args.value_loss_coef,
        entropy_coef = args.entropy_loss_coef,
        max_grad_norm = 0.5,
        num_epochs = 2,
        clip_value_loss = args.clip_value_loss,
        huber_value_loss = False,
    ),
    pbt = pbt_cfg,
    value_normalizer_decay = 0.999,
    compute_dtype = dtype,
    seed = 5,
)

policy = make_policy(dtype)

if args.restore:
    restore_ckpt = os.path.join(
        args.ckpt_dir, args.run_name, str(args.restore))
else:
    restore_ckpt = None

try:
    madrona_learn.train(dev, cfg, sim_init, sim_step, policy, iter_cb,
        CustomMetricConfig(add_metrics = lambda metrics: metrics),
        restore_ckpt = restore_ckpt, profile_port = args.profile_port)
except:
    tb_writer.flush()
    raise

tb_writer.flush()
del sim

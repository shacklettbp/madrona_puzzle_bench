import jax
from jax import lax, random, numpy as jnp
from jax.experimental import checkify
import flax
from flax import linen as nn

import argparse
from functools import partial
from time import time
from dataclasses import dataclass
import os
import numpy as np

import madrona_puzzle_bench
from madrona_puzzle_bench import SimFlags, RewardMode
from madrona_puzzle_bench.madrona import ExecMode

import madrona_learn
from madrona_learn import (
    TrainConfig, TrainHooks, PPOConfig, PBTConfig, ParamExplore,
    TensorboardWriter, WandbWriter
)
import wandb

from jax_policy import make_policy, actions_config

madrona_learn.cfg_jax_mem(0.6)

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
arg_parser.add_argument('--minibatch-size', type=int, default=4096)

arg_parser.add_argument('--lr', type=float, default=1e-4)
arg_parser.add_argument('--gamma', type=float, default=0.998)
arg_parser.add_argument('--entropy-loss-coef', type=float, default=0.01)
arg_parser.add_argument('--value-loss-coef', type=float, default=0.5)
arg_parser.add_argument('--clip-value-loss', action='store_true')

arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')
arg_parser.add_argument('--bf16', action='store_true')

arg_parser.add_argument('--eval-frequency', type=int, default=500)
arg_parser.add_argument('--pbt-ensemble-size', type=int, default=0)

arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--profile-port', type=int, default=None)
arg_parser.add_argument('--wandb', action='store_true')

# Variance args
arg_parser.add_argument('--num-env-copies', type=int, default=1)

args = arg_parser.parse_args()

sim = madrona_puzzle_bench.SimManager(
    exec_mode = ExecMode.CUDA if args.gpu_sim else ExecMode.CPU,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    num_pbt_policies = args.pbt_ensemble_size,
    rand_seed = 5,
    sim_flags = SimFlags.Default,
    reward_mode = RewardMode.Dense1,
    episode_len = 200,
    levels_per_episode = 1,
    button_width = 1.5,
    door_width = 20.0 / 3.,
    reward_per_dist = 0.05,
    slack_reward = -0.005,
)

jax_gpu = jax.devices()[0].platform == 'gpu'

sim_fns = sim.jax(jax_gpu)
if args.wandb:
    tb_writer = WandbWriter(os.path.join(args.tb_dir, args.run_name), args=args)
else:
    tb_writer = TensorboardWriter(os.path.join(args.tb_dir, args.run_name))

last_time = 0
last_update = 0

def _host_cb(training_mgr):
    global last_time, last_update

    update_id = int(training_mgr.update_idx)

    cur_time = time()
    update_diff = update_id - last_update

    print(f"Update: {update_id}")
    if last_time != 0:
        print("  FPS:", args.num_worlds * args.steps_per_update * update_diff / (cur_time - last_time))

    last_time = cur_time
    last_update = update_id

    lrs = training_mgr.state.train_states.hyper_params.lr
    #entropy_coefs = training_mgr.train_states.hyper_params.entropy_coef
    #reward_hyper_params = training_mgr.policy_states.reward_hyper_params

    old_printopts = np.get_printoptions()
    np.set_printoptions(formatter={'float_kind':'{:.1e}'.format}, linewidth=150)

    if args.pbt_ensemble_size > 0:
        print(lrs)
        #print(entropy_coefs)
        #print(reward_hyper_params[..., 0])

    episode_scores = training_mgr.state.policy_states.episode_score.mean
    print(episode_scores)
    np.set_printoptions(**old_printopts)

    print()

    training_mgr.log_metrics_tensorboard(tb_writer)

    for i in range(episode_scores.shape[0]):
        tb_writer.scalar(f"p{i}/episode_score", episode_scores[i], update_id)

    #if args.pbt_ensemble_size > 0:
    #    for i in range(episode_scores.shape[0]):
    #        tb_writer.scalar(f"p{i}/dist_to_exit_scale",
    #                         reward_hyper_params[i][0], update_id)

    num_train_policies = lrs.shape[0]
    for i in range(lrs.shape[0]):
        tb_writer.scalar(f"p{i}/lr", lrs[i], update_id)
        #tb_writer.scalar(f"p{i}/entropy_coef", entropy_coefs[i], update_id)

    return ()

def start_rollouts(self, rollout_state, user_state):
    return rollout_state, user_state
    #ckpts = rollout_state.get_current_checkpoints()
    #main_ckpts = ckpts.reshape(-1, args.num_env_copies, ckpts.shape[-1])
    #main_ckpts = main_ckpts[:, 0]
    #ckpts = jnp.repeat(main_ckpts, args.num_env_copies, axis=0)

    #return rollout_state.load_checkpoints_into_sim(ckpts), user_state

dev = jax.devices()[0]

if args.pbt_ensemble_size != 0:
    pbt_cfg = PBTConfig(
        num_teams = 1,
        team_size = 1,
        num_train_policies = args.pbt_ensemble_size,
        num_past_policies = 0,
        self_play_portion = 1.0,
        cross_play_portion = 0.0,
        past_play_portion = 0.0,
        #reward_hyper_params_explore = {
        #    'dist_to_exit_scale': ParamExplore(
        #        base = 0.05,
        #        min_scale = 0.1,
        #        max_scale = 10.0,
        #        log10_scale = True,
        #    ),
        #}
    )
else:
    pbt_cfg = None

if args.fp16:
    dtype = jnp.float16
elif args.bf16:
    dtype = jnp.bfloat16
else:
    dtype = jnp.float32


if pbt_cfg:
    lr = ParamExplore(
        base = args.lr,
        min_scale = 0.1,
        max_scale = 10.0,
        log10_scale = True,
    )

    #entropy_coef = ParamExplore(
    #    base = args.entropy_loss_coef,
    #    min_scale = 0.1,
    #    max_scale = 10.0,
    #    log10_scale = True,
    #)
else:
    lr = args.lr
    entropy_coef = args.entropy_loss_coef


cfg = TrainConfig(
    num_worlds = args.num_worlds,
    num_agents_per_world = 1,
    num_updates = args.num_updates,
    actions = actions_config,
    steps_per_update = args.steps_per_update,
    metrics_buffer_size = 10,
    num_bptt_chunks = args.num_bptt_chunks,
    lr = lr,
    gamma = args.gamma,
    gae_lambda = 0.95,
    algo = PPOConfig(
        minibatch_size = args.minibatch_size,
        clip_coef = 0.2,
        value_loss_coef = args.value_loss_coef,
        entropy_coef = {
            'act': args.entropy_loss_coef,
        },
        max_grad_norm = 0.5,
        num_epochs = 2,
        clip_value_loss = args.clip_value_loss,
        huber_value_loss = False,
    ),
    pbt = pbt_cfg,
    dreamer_v3_critic = False,
    hlgauss_critic = True,
    normalize_values = False,
    value_normalizer_decay = 0.999,
    compute_dtype = dtype,
    filter_advantages = False,
    importance_sample_trajectories = False,
    seed = 5,
)

policy = make_policy(dtype)

if args.restore:
    restore_ckpt = os.path.join(
        args.ckpt_dir, args.run_name, str(args.restore))
else:
    restore_ckpt = None


def update_loop(training_mgr):
    assert args.eval_frequency % cfg.metrics_buffer_size == 0

    def inner_iter(i, training_mgr):
        return training_mgr.update_iter()

    def outer_iter(i, training_mgr):
        training_mgr = lax.fori_loop(
            0, cfg.metrics_buffer_size, inner_iter, training_mgr)

        jax.experimental.io_callback(
            _host_cb, (), training_mgr, ordered=True)

        return training_mgr

    return lax.fori_loop(0, args.eval_frequency // cfg.metrics_buffer_size,
                         outer_iter, training_mgr)


try:
    training_mgr = madrona_learn.init_training(dev, cfg, sim_fns, policy,
        init_sim_ctrl=jnp.array([0], jnp.int32),
        restore_ckpt=restore_ckpt,
        profile_port=args.profile_port)

    assert training_mgr.update_idx % args.eval_frequency == 0
    num_outer_iters = ((args.num_updates - int(training_mgr.update_idx)) //
        args.eval_frequency)

    update_loop_compiled = madrona_learn.aot_compile(update_loop, training_mgr)

    #update_population_compiled = madrona_learn.aot_compile(update_population, training_mgr)

    last_time = time()

    for i in range(num_outer_iters):
        training_mgr = update_loop_compiled(training_mgr)

        #training_mgr = update_population_compiled(training_mgr)

        training_mgr.save_ckpt(f"{args.ckpt_dir}/{args.run_name}")
    
    madrona_learn.stop_training(training_mgr)


except:
    tb_writer.flush()
    raise

tb_writer.flush()
del sim

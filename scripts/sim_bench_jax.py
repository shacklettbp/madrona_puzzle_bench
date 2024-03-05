import jax
from jax import lax, random, numpy as jnp
import madrona_puzzle_bench
from madrona_puzzle_bench import RewardMode
import argparse
import time

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--gpu-id', type=int, default=0)

args = arg_parser.parse_args()

reward_mode = getattr(RewardMode, "Dense1")

sim = madrona_puzzle_bench.SimManager(
    exec_mode = madrona_puzzle_bench.madrona.ExecMode.CUDA,
    gpu_id = args.gpu_id,
    num_worlds = args.num_worlds,
    rand_seed = 5,
    sim_flags = (int)(0),
    reward_mode = reward_mode,
    episode_len = 200,
    levels_per_episode = 1,
    button_width = 1.3,
    door_width = 20.0 / 3.,
    reward_per_dist = 0.05,
    slack_reward = -0.005,
    num_pbt_policies = 0,
)

sim_init, sim_step = sim.jax(True)

sim_init()

@jax.jit
def f():
    def loop(i, rnd):
        rnds = random.split(rnd, 5)

        actions = jnp.stack([
                random.randint(rnds[1], (args.num_worlds,), 0, 4, jnp.int32),
                random.randint(rnds[2], (args.num_worlds,), 0, 8, jnp.int32),
                random.randint(rnds[3], (args.num_worlds,), 0, 5, jnp.int32),
                random.randint(rnds[4], (args.num_worlds,), 0, 2, jnp.int32),
            ], axis=-1)

        resets = jnp.zeros((args.num_worlds,), jnp.int32)
        
        sim_step({
            'actions': actions,
            'resets': resets,
        })

        return rnds[0]

    lax.fori_loop(0, args.num_steps, loop, random.key(5))

compiled_f = f.lower().compile()

start = time.time()
compiled_f()
end = time.time()

print("FPS", args.num_steps * args.num_worlds / (end - start))

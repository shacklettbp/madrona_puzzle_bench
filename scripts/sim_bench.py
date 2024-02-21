import torch
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
)

actions = sim.action_tensor().to_torch()
checkpoints = sim.checkpoint_tensor().to_torch()
checkpoint_resets = sim.checkpoint_reset_tensor().to_torch()
resets = sim.reset_tensor().to_torch()

start = time.time()
for i in range(args.num_steps):
    # Rotate worlds
    checkpoint_resets[:, 0] = 1
    # Shift all tensors in checkpoints by one position
    checkpoints[1:] = checkpoints[:-1].clone()
    sim.step()

    # Now take an action on all worlds
    actions[..., 0] = torch.randint_like(actions[..., 0], 0, 4)
    actions[..., 1] = torch.randint_like(actions[..., 1], 0, 8)
    actions[..., 2] = torch.randint_like(actions[..., 2], 0, 5)
    actions[..., 3] = torch.randint_like(actions[..., 3], 0, 2)

    sim.step()

end = time.time()

print("FPS", args.num_steps * args.num_worlds / (end - start))

import torch
import madrona_puzzle_bench
from madrona_puzzle_bench import SimFlags, RewardMode
from madrona_puzzle_bench_learn import LearningState

import time

from policy import make_policy, setup_obs

import numpy as np
import argparse
import math
from pathlib import Path
import warnings
warnings.filterwarnings("error")

import os
import json

import obby
from obby.examples.example_generator import *
from obby.examples.distribution_generators import *

torch.manual_seed(0)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--gpu-id', type=int, default=0)
arg_parser.add_argument('--ckpt-path', type=str, required=True)
arg_parser.add_argument('--record-log', type=str)
arg_parser.add_argument('--trajectory-log', type=str)
arg_parser.add_argument('--action-replay', type=str, help="Binary with a tensor of actions to take along with starting position.")

arg_parser.add_argument('--seed', type=int, default=10)


arg_parser.add_argument('--use-fixed-world', action='store_true')

arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-trials', type=int, required=True)
arg_parser.add_argument('--no-level-obs', action='store_true')

arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')

arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--use-complex-level', action='store_true')

# Architecture args
arg_parser.add_argument('--entity-network', action='store_true')

arg_parser.add_argument('--json-levels', type=str, default="", help="JSON level description or folder of JSON level descriptions.")
arg_parser.add_argument('--write-json', action='store_true', help="Write a binary .out file from the json file and exit.")

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
print("started init")

sim.init()

print("finished init")

action_dims = [2, 8, 1, 2]

obs, num_obs_features = setup_obs(sim, args.no_level_obs)
policy = make_policy(num_obs_features, None, args.num_channels, args.separate_value, action_dims)

json_indices = sim.json_index_tensor().to_torch()
json_levels = sim.json_level_descriptions_tensor().to_torch()
resets = sim.reset_tensor().to_torch()

print(json_levels.shape)

SPAWN_TYPE = 14 # new "Spawn" type, just signals to make the spawn restraints there.
CHECKPOINT_TYPE = 13 # Corresponds to "Goal" in types.hpp
WALL_TYPE = 9
LAVA_TYPE = 5

def jsonTableToTensor(jsonLevel):
    #print(jsonLevel)
    # Single level has max 64 objects, 7 floats/object
    out_tensor = torch.zeros(64, 10)
    idx = 0
    unit = 1
    for name in jsonLevel.keys():
        objDesc = jsonLevel[name]

        objType = WALL_TYPE

        for behaviorDict in objDesc["behaviors"]:
            objTypeName = behaviorDict["name"]
            if "Spawn" in objTypeName:
                objType = SPAWN_TYPE
            elif "Goal" in objTypeName or "Checkpoint" in objTypeName:
                objType = CHECKPOINT_TYPE
            elif "Damage" in objTypeName:
                objType = LAVA_TYPE

        #if len(objDesc["tags"]) != 0:
        #    # TODO: only process normal walls.
        #    continue
        out_tensor[idx][:3] = torch.tensor(objDesc["position"])
        out_tensor[idx][3:6] = torch.tensor(objDesc["size"])
        out_tensor[idx][6:9] = torch.tensor(objDesc["orientation"])
        out_tensor[idx][9] = objType
        idx += 1
    return out_tensor

def jsonFileToTensor(jsonFile):
    jsonLevel = []
    with open(jsonFile, "r") as f:
        jsonString = f.read()
        jsonLevel = json.loads(jsonString)
        return jsonTableToTensor(jsonLevel)

jsonLevelList = [
    "path_jump_path",
    "test_platform",
    "lilypads",
    "lava_tightrope",
    "lava_pattern",
    "lava_checkerboard"
]


n_blocks = UniformInt(2, 2, seed=args.seed)
gap_length = Uniform(1, 1, seed=args.seed)
#gap_length = Uniform(0, 2, seed=10)
block_length = Uniform(1.0, 6.0, seed=args.seed)
for i in range(1024):
    json_levels[i] = jsonTableToTensor(json.loads(random_path_jump_path_generator(f"path_jump_path_s{args.seed}_{i}", n_blocks, gap_length, block_length).jsonify()))

for i in range(args.num_worlds):
    json_indices[i, 0] = i


# Read JSON levels from a file
#if args.json_levels != "":
#    if os.path.isfile(args.json_levels):
#        json_levels[0] = jsonFileToTensor(args.json_levels)
#    else:
#        levelIdx = 0
#        for jsonFile in os.path.listdir(args.json_levels):
#            if jsonFile.endswith(".json"):
#                json_levels[levelIdx] = jsonFileToTensor(os.path.join(args.json_levels, jsonFile))
#                levelIdx += 1

if args.write_json:
    print(json_levels[0])
    outfile = args.json_levels + ".out"
    print(outfile)
    with open(outfile, "wb") as f:
        f.write(json_levels.numpy().tobytes())


#json_indices[:, 0] = 0

# print(json_levels.shape)
# print("Testing")
# # TODO: restore, test export.
# for i in range(64):
#     jsonObj = json_levels[0][i]
#     jsonObj[:3] = torch.tensor([0,0,0])
#     jsonObj[3:6] = torch.tensor([1,1,1])
#     jsonObj[6] = 0 if i > 0 else 9



# resets[:, 0] = 1
# print("Wrote resets")

# print("done writing")

# for i in range(3):
#     resets[:, 0] = 1
#     json_indices[:, 0] = i
#     sim.step()
#TODO: 
# 0. Figure out how to switch the goal dynamically within the world.
# 1. Load multiple sets of weights for different low level controllers.
# 2. Detect when the first controller is done to be able to easily change to the second controller.
#    Actually, as long as we're using the right reward model, we should be able to tell because the 
#    reward will be 0 until we've achieved the subgoal.

weights = LearningState.load_policy_weights(args.ckpt_path)
policy.load_state_dict(weights, strict=False)

policy = policy.to(torch.device(f"cuda:{args.gpu_id}")) if args.gpu_sim else policy.to(torch.device('cpu'))

actions = sim.action_tensor().to_torch()
dones = sim.done_tensor().to_torch()
rewards = sim.reward_tensor().to_torch()
goals = sim.goal_tensor().to_torch()

ckpts = sim.checkpoint_tensor().to_torch()

cur_rnn_states = []

for shape in policy.recurrent_cfg.shapes:
    cur_rnn_states.append(torch.zeros(
        *shape[0:2], actions.shape[0], shape[2], dtype=torch.float32, device=torch.device('cpu')))

if args.record_log:
    record_log = open(args.record_log, 'wb')
else:
    record_log = None


if args.trajectory_log:
    trajectory_log = open(args.trajectory_log, 'w')
else:
    trajectory_log = None


count = 0
timings = 0

trajectory_step = {
    "observations" : {},
    "action" : {},
    "action_probs" : {},
    "worldIdx" : -1
}

trajectories = []

remainingTrials = args.num_trials
successes = 0
while True:


    #print("Observations")
    #for o in obs:
    #    print(o)

    #print(obs[-3]) # Steps remaining

    #print(obs[0][..., :3])

    trajectory_step["observations"] = [o.tolist() for o in obs]

    start = time.time()
    with torch.no_grad():
        action_dists, values, cur_rnn_states = policy(cur_rnn_states, *obs)
        #action_dists.best(actions)
        # Make placeholders for actions_out and log_probs_out
        if True:
            log_probs_out = torch.zeros_like(actions).float()
            action_dists.sample(actions, log_probs_out)
        else:
            action_dists.best(actions)

        trajectory_step["action"] = actions.tolist()
        probs = action_dists.probs()
        trajectory_step["action_probs"] = [p.tolist() for p in probs]
    end = time.time()

    # TODO: restore
    #actions[:] = torch.tensor([1, 0, 0, 1]).unsqueeze(0)

    # For trajectory playback we assume only world 0 is active.
    trajectory_step["worldIdx"] = int(json_indices[:, 0])

    if trajectory_log:
        trajectories.append(trajectory_step.copy())
    
    timings += end - start
    count += 1
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

    #print("Sim Step Time:", end - start)
    #print("Rewards:\n", rewards)
    if rewards[..., 0] >= 10.0:
        successes += 1
    #print("Goals:\n", goals)

    if dones[0, 0] == 1:
        remainingTrials -= 1

    if remainingTrials == 0:
        break

    # If the reward is positive, update the goal to the next Object ID
    # ONLY WORKS with sparse reward.
    # TODO: configure this to work with multiple worlds.
    #if goals[..., 1] == 1:
    #    print("SWITCHING GOAL")
    #    # We hit the goal, change the goal.
    #    if goals[..., 0] == 6:
    #        goals[..., 0] = 2
    #    elif goals[..., 0] == 2:
    #        goals[..., 0] = 0 # 0 encodes the nonetype which causes the goal marker to switch to the exit.


print("Average Time:", 1000 * timings / count, "ms")
print(f"Succeeded in {successes} / {args.num_trials}")

if record_log:
    record_log.close()

if trajectory_log:
    trajectory_log.write(json.dumps(trajectories))
    trajectory_log.close()

import torch
import madrona_puzzle_bench
from madrona_puzzle_bench import SimFlags, RewardMode
from madrona_puzzle_bench_learn import LearningState
import time

import keyboard
from flask import Flask
from flask import request
import json

# Keep flasking from printing messages on very http request
#import logging
#log = logging.getLogger('werkzeug')
#log.setLevel(logging.ERROR)

app = Flask(__name__)

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
arg_parser.add_argument('--replay-trajectories', type=str, help="JSON file of trajectories to replay.")
arg_parser.add_argument('--key-control', action='store_true', help="If true, control actions with the keyboard.")



arg_parser.add_argument('--use-fixed-world', action='store_true')

arg_parser.add_argument('--num-worlds', type=int, required=True)
arg_parser.add_argument('--num-steps', type=int, required=True)
arg_parser.add_argument('--no-level-obs', action='store_true')

arg_parser.add_argument('--num-channels', type=int, default=256)
arg_parser.add_argument('--separate-value', action='store_true')
arg_parser.add_argument('--fp16', action='store_true')

arg_parser.add_argument('--gpu-sim', action='store_true')
arg_parser.add_argument('--use-complex-level', action='store_true')

# Architecture args
arg_parser.add_argument('--entity-network', action='store_true')

args = arg_parser.parse_args()

sim_flags = madrona_puzzle_bench.SimFlags.Default
if args.use_complex_level:
    sim_flags |= madrona_puzzle_bench.SimFlags.UseComplexLevel
print(sim_flags)

# Hack to  get around to_torch() conversion in
# madrona_puzzle_bench scripts.
class ToTorchWrapper:
    def __init__(self, tensor):
        self.tensor = tensor
    def to_torch(self):
        return self.tensor
        
class RobloxSimManager:

    def __init__(self):
        # Obs tensors modified by Roblox.
        self.agent_txfm_obs = torch.zeros([1, 10])
        self.agent_exit_obs = torch.zeros([1, 3])
        self.entity_physics_state_obs = torch.zeros([1, 9, 12])
        self.entity_type_obs = torch.zeros([1, 9, 1])
        self.lidar_depth = torch.zeros([1, 30, 1])
        self.lidar_type = torch.zeros([1, 30, 1]) 
        self.steps_remaining = torch.zeros([1, 1])
        self.steps_remaining[...] = 200 # initialize steps remaining.

        # Action to pass to Roblox.
        self.action = torch.zeros([1, 4])

    def obsString(self):
        return str(self.agent_txfm_obs) + "\n" + \
        str(self.agent_exit_obs) + "\n" + \
        str(self.entity_physics_state_obs) + "\n" + \
        str(self.entity_type_obs) + "\n" + \
        str(self.lidar_depth) + "\n" + \
        str(self.lidar_type)

    # Observation tensors.
    def agent_txfm_obs_tensor(self):
        # TODO: return a tensor that matches the agent observations
        # localRoomPos, room AABB, theta (not needed).
        return ToTorchWrapper(self.agent_txfm_obs)
    
    def agent_interact_obs_tensor(self):
        # Bool, whether or not grabbing
        # Always zero in Roblox.
        return ToTorchWrapper(torch.zeros([1, 1]))
    
    def agent_level_type_obs_tensor(self):
        level_type = torch.zeros([1, 1])
        level_type[0, 0] = 10 # Always LavaCorridor
        return ToTorchWrapper(level_type)
    
    def agent_exit_obs_tensor(self): # Scalars
        # Polar exit observation, set by Roblox.
        return ToTorchWrapper(self.agent_exit_obs)
    
    def entity_physics_state_obs_tensor(self): # Scalars
        # Phyics state, set by roblox.
        return ToTorchWrapper(self.entity_physics_state_obs)
    
    def entity_type_obs_tensor(self): # Enum
        # Entity type. Will always be lava, set
        # by Roblox.
        return ToTorchWrapper(self.entity_type_obs)
    
    def entity_attr_obs_tensor(self): # Enum, but I don't know max
        # Entity attributes (e.g. door open/closed), always 0.
        return ToTorchWrapper(torch.zeros([1, 9, 2]))
    
    # Lidar shouldn't be able to see the lava.
    # For a first pass, we return 0 (no hit)
    # for all lidar samples.
    def lidar_depth_tensor(self): # Scalars
        return ToTorchWrapper(self.lidar_depth)
    def lidar_hit_type(self): # Enum (EntityType)
        return ToTorchWrapper(self.lidar_type)
    
    def steps_remaining_tensor(self): # Int but dont' need to convert
        return ToTorchWrapper(self.steps_remaining)
    
    # Action tensors
    def action_tensor(self):
        return ToTorchWrapper(self.action)
    
    # None of these matter for Roblox because the 
    # evaluation (for now) is video, but eventually
    # We will want to translate the reward signal.
    def done_tensor(self):
        return ToTorchWrapper(torch.zeros([1, 1]))
    def reward_tensor(self):
        return ToTorchWrapper(torch.zeros([1, 1]))
    def goal_tensor(self):
        return ToTorchWrapper(torch.zeros([1, 2]))
    def checkpoint_tensor(self):
        return ToTorchWrapper(torch.zeros([1, 1388]))
    
    def init(self):
        pass

# TODO: Restore
#sim = madrona_puzzle_bench.SimManager(
#    exec_mode = madrona_puzzle_bench.madrona.ExecMode.CUDA if args.gpu_sim else madrona_puzzle_bench.madrona.ExecMode.CPU,
#    gpu_id = args.gpu_id,
#    num_worlds = args.num_worlds,
#    rand_seed = 10,
#    sim_flags = (int)(sim_flags),
#    reward_mode = getattr(RewardMode, "Dense1"),
#    episode_len = 200,
#    levels_per_episode = 1,
#    button_width = 1.3,
#    door_width = 20.0 / 3.,
#    reward_per_dist = 0.05,
#    slack_reward = -0.005,
#)
#sim.init()

# TODO: restore
sim = RobloxSimManager()

obs, num_obs_features = setup_obs(sim, args.no_level_obs)

policy = make_policy(num_obs_features, None, args.num_channels, args.separate_value)

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


if args.replay_trajectories:
    with open(args.replay_trajectories, "r") as f:
        trajectories = json.loads(f.read())
    trajectory_start_indices = []
    #Convert trajectories to tensor form
    for idx, t in enumerate(trajectories):
        t["observations"] = [torch.tensor(o) for o in t["observations"]]
        t["action"] = torch.tensor(t["action"])
        t["action_probs"] = [torch.tensor(p) for p in t["action_probs"]]
        if t["observations"][-3][...] == 200:
            trajectory_start_indices.append(idx)
        current_trajectory = -1
        current_trajectory_step = 0
    print(len(trajectories))
    print(trajectory_start_indices)
    print(trajectories[0]["observations"])
else:
    trajectories = None
    trajectory_start_indices = None
    current_trajectory = None
    current_trajectory_step = None

#Analyze the first trajectory
for i in range(trajectory_start_indices[1]):
    print(trajectories[i]["action"])

print([i1 - i0 for i0,i1 in zip(trajectory_start_indices[:-1], trajectory_start_indices[1:])])


controlDict = {
    "w" : "up",
    "a" : "up",
    "s" : "up",
    "d" : "up",
    "space" : "up"
}

def xyzToPolar(v):
    #print(v)
    r = math.sqrt(sum([x*x for x in v]))

    if r < 1e-5:
        return [0,0,0]

    v = [x / r for x in v]

    # Note that this is angle off y-forward
    theta = -math.atan2(v[0], v[1])
    phi = math.asin(max(-1.0, min(1.0, v[2])))

    return[r, theta, phi]


# We ignore rotation for the moment.
# The agent shouldn't need it for this challenge.
actionJson = {
    "moveAmount" : 0,
    "moveAngle" : 0,
    "jump" : 0,
    "obsTime" : 0,
    "kill" : False,
    "startPos" : [0,0,0]
}

MADRONA_TO_ROBLOX_SCALE = 4


def processKeyboardInput():
    moveAngles = [i for i in range(8)]

    def rem(*args):
        for x in args:
            if x in moveAngles:
                moveAngles.remove(x)

    if not keyboard.is_pressed("w"):
        rem(0, 1, 7)
    if not keyboard.is_pressed("a"):
        rem(5, 6, 7)
    if not keyboard.is_pressed("s"):
        rem(3, 4, 5)
    if not keyboard.is_pressed("d"):
        rem(1, 2, 3)

    a = actionJson.copy()

    a["moveAmount"] = 0 if len(moveAngles) == 0 else 3
    a["moveAngle"] = 0 if len(moveAngles) == 0 else moveAngles[0]
    a["jump"] = 1 if keyboard.is_pressed("space") else 0

    return a


def sendAction():
    global cur_rnn_states
    global current_trajectory_step
    global current_trajectory

    # TODO: restore
    #print("Observations")
    #for o in obs:
    #    print(o)

    a = actionJson.copy()

    if trajectories:
        # Track whether to adjust the start position of the agent.
        if sim.steps_remaining[...] == 200:
            # Start a new trajectory, looping if necessary.
            current_trajectory  = (current_trajectory + 1) % len(trajectory_start_indices)
            current_trajectory_step = 0
            print("Starting Trajectory {}/{}:".format(current_trajectory,len(trajectory_start_indices)))

            # Translate new starting position back to world space (not AABB relative)
            newStartPos = trajectories[current_trajectory]["observations"][0][..., :3] + \
                (trajectories[current_trajectory]["observations"][0][..., 3:6] + trajectories[current_trajectory]["observations"][0][..., 6:9]) * 0.5
            a["startPos"] = (newStartPos.squeeze() * MADRONA_TO_ROBLOX_SCALE).tolist()
            print(a["startPos"])

        t_step_idx = trajectory_start_indices[current_trajectory] + current_trajectory_step
        print(t_step_idx)
        print(trajectory_start_indices[(current_trajectory + 1) % len(trajectory_start_indices)])
        # Account for differing action rates between madrona and roblox.
        current_trajectory_step += 2
        
        if (t_step_idx % len(trajectories)) > trajectory_start_indices[(current_trajectory + 1) % len(trajectory_start_indices)]:
            print("KILL")
            # We reached the end of this trajectory, but the character is still alive.
            # Send kill signal, which forces the character to respawn and will move us to the next trajectory.
            a["kill"] = True
        actions = trajectories[t_step_idx]["action"]
    else:
        pass
        # TODO: restore
        # Live inference with Madrona policy.
        #with torch.no_grad():
        #    action_dists, values, cur_rnn_states = policy(cur_rnn_states, *obs)
        #    #action_dists.best(actions)
        #    # Make placeholders for actions_out and log_probs_out
        #    log_probs_out = torch.zeros_like(actions).float()
        #    action_dists.sample(actions, log_probs_out)



    if args.key_control:
        a = processKeyboardInput()
    else:
        # Write policy actions (possibly from a trajectory)
        a["moveAmount"] = int(actions[..., 0])
        a["moveAngle"] = int(actions[..., 1])
        a["jump"] = int(actions[..., 3])

    sim.steps_remaining[...] -= 1

    return json.dumps(a)


@app.route("/index.json", methods=['POST'])
def receiveObservations():
    data = request.get_json()

    # Record the time.
    actionJson["obsTime"] = data["obsTime"]
    # Timing
    # global timings
    # global timeIdx
    # global hasLogged
    # global timingFrames

    # start = time.time()
    # end = time.time()
    # if timeIdx < timingFrames:
    #     obsTime = data["obsTime"] + 60 * 60 * 8
    #     recTime = time.time()
    #     timings[timeIdx] = { "timingDiff" : recTime - obsTime, "jsonTime" : end - start }
    #     timeIdx += 1
    # elif not hasLogged:
    #     with open("timinglogLimit.txt", "a") as f:
    #         f.write(str(timings))
    #     print("wrote file")
    #     hasLogged = True
    # else:
    #     print("done")
    # Update the observation tensors.

    #for idx, ob in enumerate(data["lidar"]):
    #    print(idx, ob)
    #for key in data.keys():
    #    print(key, ":", data[key])

    #print("Sent Time:", obsTime, "Rec Time:", recTime, "Difference:", recTime - obsTime)

    pos = data["playerPos"]
    # Update agent observations
    # Agent position, not polar.
    for i in range(9):
        if i < 3:
            sim.agent_txfm_obs[..., i] = pos[i] / MADRONA_TO_ROBLOX_SCALE
        elif i < 9:
            sim.agent_txfm_obs[..., i] = data["roomAABB"][(i - 3) // 3][i % 3] / MADRONA_TO_ROBLOX_SCALE
    sim.agent_txfm_obs[..., :3] -= (sim.agent_txfm_obs[..., 3:6] + sim.agent_txfm_obs[..., 6:9]) * 0.5
    # Theta
    sim.agent_txfm_obs[..., 9] = 0

    #print(sim.agent_txfm_obs)

    # Agent relative exit vector, polar.
    # Note we are correctly discarding agent rotation
    # which hopefully the network is robust to.
    for i in range(3):
        sim.agent_exit_obs[..., i] = data["goalPos"][i] / MADRONA_TO_ROBLOX_SCALE
    sim.agent_exit_obs -= torch.tensor(data["playerPos"]) / MADRONA_TO_ROBLOX_SCALE
    sim.agent_exit_obs[..., 2] = 0 #
    #print("pre XYZ", sim.agent_exit_obs)
    for i, x in enumerate(xyzToPolar(sim.agent_exit_obs.squeeze().tolist())):
        sim.agent_exit_obs[..., i] = x

    #sim.entity_physics_state_obs = torch.zeros([1, 9, 12])

    # Update physics state and entity type observations.
    # Lava is type 5
    #print("NumLava:", len(data["lava"]))
    for idx, lava in enumerate(data["lava"]):
        positionPolar = xyzToPolar([x - y for x, y in zip(lava[:3], pos)])
        # Physics state update
        for i in range(12):
            if i < 3:
                # Position
                sim.entity_physics_state_obs[0, idx, i] = positionPolar[i] / MADRONA_TO_ROBLOX_SCALE
            elif i < 6:
                # Velocity
                sim.entity_physics_state_obs[0, idx, i] = 0 # velocity.
            elif i < 9:
                # Extents
                sim.entity_physics_state_obs[0, idx, i] = lava[i - 3] / MADRONA_TO_ROBLOX_SCALE
            else:
                # Rotation
                sim.entity_physics_state_obs[0, idx, i] = 0
        # 5 is the entity type for lava.
        sim.entity_type_obs[0, idx, 0] = 5

    if not data["alive"]:
        # Don't execute an action.
        sim.steps_remaining[...] = 200
        print("Dead")
        return json.dumps(actionJson)
    #print(sim.obsString())

    # TODO: restore, no LIDAR


    # Translate lidar observations.
    #for idx, ob in enumerate(data["lidar"]):
    #    sim.lidar_depth[0, idx, 0] = float(ob[0])
    #    # Walls have type 9. In this sort of challenge, they are the only thing ever hit and they are always hit.
    #    sim.lidar_type[0, idx, 0] = 9
    
    # Roblox observations.
	# local obs = {
	# 	roomAABB = AABB,
	# 	playerPos = {posZUp.X, posZUp.Y, posZUp.Z},
    #     goalPos = {goalPosZUp.X, goalPosZUp.Y, goalPosZUp.Z},
    #     lava = lavaObservations()
	# }
    return sendAction()


# TODO: debuggin.
# @app.route("/index.json", methods=['GET'])
# def sendKeypress():
#     #return "{\"key\": \"d\"}"
#     for k in controlDict.keys():
#         controlDict[k] = "down" if keyboard.is_pressed(k) else "up"
#     #shouldStep = "False"
#     #for k in controlDict.keys():
#     #    if controlDict[k] == "down":
#     #        shouldStep = "True"
#     #sendDict = controlDict.copy()
#     #sendDict["shouldStep"] = shouldStep
#     return json.dumps(controlDict)

app.run()
# TODO: restore after testing server connection.

# for i in range(args.num_steps):
#     with torch.no_grad():
#         action_dists, values, cur_rnn_states = policy(cur_rnn_states, *obs)
#         #action_dists.best(actions)
#         # Make placeholders for actions_out and log_probs_out
#         if True:
#             log_probs_out = torch.zeros_like(actions).float()
#             action_dists.sample(actions, log_probs_out)
#         else:
#             action_dists.best(actions)

#         probs = action_dists.probs()

#     if record_log:
#         ckpts.cpu().numpy().tofile(record_log)

#     '''
#     print()
#     print("Self:", obs[0])
#     print("Partners:", obs[1])
#     print("Room Entities:", obs[2])
#     print("Lidar:", obs[3])

#     print("Move Amount Probs")
#     print(" ", np.array_str(probs[0][0].cpu().numpy(), precision=2, suppress_small=True))
#     print(" ", np.array_str(probs[0][1].cpu().numpy(), precision=2, suppress_small=True))

#     print("Move Angle Probs")
#     print(" ", np.array_str(probs[1][0].cpu().numpy(), precision=2, suppress_small=True))
#     print(" ", np.array_str(probs[1][1].cpu().numpy(), precision=2, suppress_small=True))

#     print("Rotate Probs")
#     print(" ", np.array_str(probs[2][0].cpu().numpy(), precision=2, suppress_small=True))
#     print(" ", np.array_str(probs[2][1].cpu().numpy(), precision=2, suppress_small=True))

#     print("Grab Probs")
#     print(" ", np.array_str(probs[3][0].cpu().numpy(), precision=2, suppress_small=True))
#     print(" ", np.array_str(probs[3][1].cpu().numpy(), precision=2, suppress_small=True))

#     print("Actions:\n", actions.cpu().numpy())
#     print("Values:\n", values.cpu().numpy())
#     '''
#     sim.step()
#     print("Rewards:\n", rewards)
#     print("Goals:\n", goals)



if record_log:
    record_log.close()

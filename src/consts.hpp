#pragma once

#include <madrona/types.hpp>

namespace madPuzzle {

namespace consts {
// Each random world is composed of a fixed number of rooms that the agents
// must solve in order to maximize their reward.
inline constexpr madrona::CountT maxRooms = 3; // 8;

// Generated levels assume 2 agents
inline constexpr madrona::CountT numAgents = 2;

// Maximum number of interactive objects per challenge room. This is needed
// in order to setup the fixed-size learning tensors appropriately.
inline constexpr madrona::CountT maxEntitiesPerRoom = 6;//20;

// Give the agents more observation space than the number of room
// entities in case they push cubes into other rooms.
inline constexpr madrona::CountT maxObservationsPerAgent = 9;//20;

// Various world / entity size parameters
inline constexpr float worldLength = 40.0f;
inline constexpr float worldWidth = 20.f;
inline float doorWidth = worldWidth / 3.f; // Maybe should keep inline and remove extern...
inline constexpr float wallWidth = 1.f;
inline float buttonWidth = 1.3f;
inline constexpr float keyWidth = 0.5f;
inline constexpr float agentRadius = 1.f;
inline constexpr float roomLength = worldLength / 3.f;

inline constexpr int doorsPerRoom = 1;//4;
inline constexpr int wallsPerRoom = 2;//8;

// Each unit of distance forward (+ y axis) rewards the agents by this amount
inline float rewardPerDist = 0.05f;
// Each step that the agents don't make additional progress they get a small
// penalty reward
inline float slackReward = -0.005f;

// Steps per episode
inline constexpr int32_t episodeLen = 200;

// How many discrete options for actions
inline constexpr madrona::CountT numMoveAmountBuckets = 4;
inline constexpr madrona::CountT numMoveAngleBuckets = 8;
inline constexpr madrona::CountT numTurnBuckets = 5;

// Number of lidar samples, arranged in circle around agent
inline constexpr madrona::CountT numLidarSamples = 30;

// Time (seconds) per step
inline constexpr float deltaT = 0.04f;

// Speed at which doors raise and lower
inline constexpr float doorSpeed = 30.f;

// Number of physics substeps
inline constexpr madrona::CountT numPhysicsSubsteps = 4.f;

}

}

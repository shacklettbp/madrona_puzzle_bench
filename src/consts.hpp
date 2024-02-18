#pragma once

#include <madrona/types.hpp>

namespace madPuzzle {
using CountT = madrona::CountT;

namespace consts {
// Give the agents more observation space than the number of room
// entities in case they push cubes into other rooms.
inline constexpr CountT maxObservationsPerAgent = 9;
inline constexpr CountT maxObjectsPerLevel = 20;

// Various world / entity size parameters
inline constexpr float wallWidth = 1.f;

inline constexpr float worldLength = 40.0f;
inline constexpr float worldWidth = 20.f;
inline constexpr float keyWidth = 0.5f;
inline constexpr float agentRadius = 1.f;
inline constexpr float roomLength = worldLength / 3.f;

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

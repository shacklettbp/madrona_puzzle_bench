#pragma once

#include <madrona/types.hpp>

namespace madPuzzle {
using CountT = madrona::CountT;

namespace consts {


// JSON level constants
inline constexpr CountT maxJsonLevelDescriptions = 1024;
inline constexpr CountT maxJsonObjects = 64; // includes walls.

inline constexpr bool disableLidar = true;


// Give the agents more observation space than the number of room
// entities in case they push cubes into other rooms.
// 72 is the max for the current training with 8192 worlds.
inline constexpr CountT maxObservationsPerAgent = 36; //9;
inline constexpr CountT maxObjectsPerLevel = 20;

// Various world / entity size parameters
inline constexpr float wallWidth = 1.f;

inline constexpr float gravity = 196.2; // 9.8f

inline constexpr float worldLength = 40.0f;
inline constexpr float worldWidth = 20.f;
inline constexpr float keyWidth = 0.5f;
inline constexpr float agentRadius = 2.55f;
// TODO: restore
//inline constexpr madrona::math::Vector3 agentExtents = {0.5f, 0.5f, 5.1f};
inline constexpr madrona::math::Vector3 agentExtents = {4.0f, 1.2f, 5.10f};
inline constexpr float roomLength = worldLength / 3.f;

// How many discrete options for actions
inline constexpr madrona::CountT numMoveAmountBuckets = 2;
inline constexpr madrona::CountT numMoveAngleBuckets = 8;
inline constexpr madrona::CountT numTurnBuckets = 1;

// Number of lidar samples, arranged in circle around agent
inline constexpr madrona::CountT numLidarSamples = 30;

// Time (seconds) per step
inline constexpr float deltaT = 0.06f;
inline constexpr float minDeltaT = 0.02f;

// Speed at which doors raise and lower
inline constexpr float doorSpeed = 30.f;

// Number of physics substeps
inline constexpr madrona::CountT numPhysicsSubsteps = 4.f;

}

}

#pragma once

#include "sim.hpp"

namespace madPuzzle {

Entity createAgent(Engine &ctx);
void resetAgent(Engine &ctx,
                Vector3 spawn_pos,
                float spawn_size,
                Vector3 exit_pos);

LevelType generateLevel(Engine &ctx);
void destroyLevel(Engine &ctx);

}

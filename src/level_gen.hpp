#pragma once

#include "sim.hpp"

namespace madPuzzle {

Entity createAgent(Engine &ctx);

LevelType generateLevel(Engine &ctx);
void destroyLevel(Engine &ctx);

}

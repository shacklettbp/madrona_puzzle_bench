#pragma once

#include "sim.hpp"

namespace madPuzzle {

// Creates agents, outer walls and floor. Entities that will persist across
// all episodes.
void createPersistentEntities(Engine &ctx);

void generateLevel(Engine &ctx);
void destroyLevel(Engine &ctx);

}

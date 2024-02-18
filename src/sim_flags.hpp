#pragma once

#include <cstdint>

namespace madPuzzle {

enum class SimFlags : uint32_t {
    Default                = 0,
    UseFixedWorld          = 1 << 0,
    IgnoreEpisodeLength    = 1 << 1,
};

enum class RewardMode : uint32_t {
    Dense1,
    PerLevel,
    EndOnly,
};

inline SimFlags & operator|=(SimFlags &a, SimFlags b);
inline SimFlags operator|(SimFlags a, SimFlags b);
inline SimFlags & operator&=(SimFlags &a, SimFlags b);
inline SimFlags operator&(SimFlags a, SimFlags b);

}

#include "sim_flags.inl"

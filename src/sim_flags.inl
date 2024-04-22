namespace madPuzzle {

inline SimFlags & operator|=(SimFlags &a, SimFlags b)
{
    a = SimFlags(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
    return a;
}

inline SimFlags operator|(SimFlags a, SimFlags b)
{
    a |= b;

    return a;
}

inline SimFlags & operator&=(SimFlags &a, SimFlags b)
{
    a = SimFlags(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
    return a;
}

inline SimFlags operator&(SimFlags a, SimFlags b)
{
    a &= b;

    return a;
}

constexpr uint32_t levelIdMask = 0x1F;
constexpr uint32_t levelIdShift = 2; // Number of bits to shift for LevelType

// Operations for LevelType
inline LevelType extractLevelType(SimFlags flags) {
    return static_cast<LevelType>((static_cast<uint32_t>(flags) >> levelIdShift) & 0x1F);
}

inline SimFlags operator|(SimFlags lhs, LevelType rhs) {
    return static_cast<SimFlags>(static_cast<uint32_t>(lhs) | (static_cast<uint32_t>(rhs) << levelIdShift));
}

}

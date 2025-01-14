#pragma once

#include <pbrt/gpu/macro.h>
#include <pbrt/util/basic_math.h>

// Random Number Declarations
#define PCG32_DEFAULT_STATE 0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT 0x5851f42d4c957f2dULL

// Hashing Inline Functions
// http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
PBRT_CPU_GPU inline uint64_t MixBits(uint64_t v);

inline uint64_t MixBits(uint64_t v) {
    v ^= (v >> 31);
    v *= 0x7fb5d329728ea185;
    v ^= (v >> 27);
    v *= 0x81dadef4bc2dd44d;
    v ^= (v >> 33);
    return v;
}

// RNG Definition
class RNG {
  public:
    // RNG Public Methods
    PBRT_CPU_GPU
    RNG() : state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM) {}

    PBRT_CPU_GPU
    RNG(uint64_t seqIndex, uint64_t offset) {
        set_sequence(seqIndex, offset);
    }

    PBRT_CPU_GPU
    RNG(uint64_t seqIndex) {
        set_sequence(seqIndex);
    }

    PBRT_CPU_GPU
    void set_sequence(uint64_t sequenceIndex, uint64_t offset);

    PBRT_CPU_GPU
    void set_sequence(uint64_t sequenceIndex) {
        set_sequence(sequenceIndex, MixBits(sequenceIndex));
    }

    template <typename T>
    PBRT_CPU_GPU T uniform();

    template <typename T>
    PBRT_CPU_GPU typename std::enable_if_t<std::is_integral_v<T>, T> uniform(T b) {
        T threshold = (~b + 1u) % b;
        while (true) {
            T r = uniform<T>();
            if (r >= threshold)
                return r % b;
        }
    }

    PBRT_CPU_GPU
    void advance(int64_t idelta);

  private:
    // RNG Private Members
    uint64_t state, inc;
};

// RNG Inline Method Definitions
template <typename T>
inline T RNG::uniform() {
    return T::unimplemented;
}

template <>
inline uint32_t RNG::uniform<uint32_t>();

template <>
inline uint32_t RNG::uniform<uint32_t>() {
    uint64_t oldstate = state;
    state = oldstate * PCG32_MULT + inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
}

template <>
inline uint64_t RNG::uniform<uint64_t>() {
    uint64_t v0 = uniform<uint32_t>(), v1 = uniform<uint32_t>();
    return (v0 << 32) | v1;
}

template <>
inline int32_t RNG::uniform<int32_t>() {
    // https://stackoverflow.com/a/13208789
    uint32_t v = uniform<uint32_t>();
    if (v <= (uint32_t)std::numeric_limits<int32_t>::max()) {
        return int32_t(v);
    }

    return int32_t(v - std::numeric_limits<int32_t>::min()) + std::numeric_limits<int32_t>::min();
}

template <>
inline int64_t RNG::uniform<int64_t>() {
    // https://stackoverflow.com/a/13208789
    uint64_t v = uniform<uint64_t>();
    if (v <= (uint64_t)std::numeric_limits<int64_t>::max()) {
        // Safe to type convert directly.
        return int64_t(v);
    }

    return int64_t(v - std::numeric_limits<int64_t>::min()) + std::numeric_limits<int64_t>::min();
}

inline void RNG::set_sequence(uint64_t sequenceIndex, uint64_t offset) {
    state = 0u;
    inc = (sequenceIndex << 1u) | 1u;
    uniform<uint32_t>();
    state += offset;
    uniform<uint32_t>();
}

template <>
inline float RNG::uniform<float>() {
    return std::min<float>(OneMinusEpsilon, uniform<uint32_t>() * 0x1p-32f);
}

template <>
inline double RNG::uniform<double>() {
    return std::min<double>(OneMinusEpsilon, uniform<uint64_t>() * 0x1p-64);
}

inline void RNG::advance(int64_t idelta) {
    uint64_t curMult = PCG32_MULT, curPlus = inc, accMult = 1u;
    uint64_t accPlus = 0u, delta = (uint64_t)idelta;
    while (delta > 0) {
        if (delta & 1) {
            accMult *= curMult;
            accPlus = accPlus * curMult + curPlus;
        }
        curPlus = (curMult + 1) * curPlus;
        curMult *= curMult;
        delta /= 2;
    }
    state = accMult * state + accPlus;
}

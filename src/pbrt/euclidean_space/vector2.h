#pragma once

#include <stdexcept>
#include "pbrt/util/macro.h"

template <typename T>
class Vector2 {
  public:
    T x, y;

    PBRT_CPU_GPU Vector2(T _x, T _y) : x(_x), y(_y) {}

    PBRT_CPU_GPU T &operator[](uint8_t index) {
        switch (index) {
        case 0: {
            return x;
        }
        case 1: {
            return y;
        }
        default: {
#if defined(__CUDA_ARCH__)
            asm("trap;");
#else
            throw std::runtime_error("Vector2: invalid index `" + std::to_string(index) + "`");
#endif
        }
        }
    }
};

using Vector2f = Vector2<FloatType>;

#pragma once

#include <pbrt/gpu/macro.h>

template <typename T>
class Vector2 {
  public:
    T x, y;

    PBRT_CPU_GPU Vector2(T _x, T _y) : x(_x), y(_y) {}

    PBRT_CPU_GPU
    Vector2 operator-() const {
        return Vector2(-x, -y);
    }

    PBRT_CPU_GPU
    Vector2 operator-(const Vector2 &v) const {
        return Vector2(x - v.x, y - v.y);
    }

    PBRT_CPU_GPU
    T &operator[](uint8_t index) {
        switch (index) {
        case 0: {
            return x;
        }
        case 1: {
            return y;
        }
        }

        REPORT_FATAL_ERROR();
        return x;
    }

    PBRT_CPU_GPU
    T operator[](uint8_t index) const {
        switch (index) {
        case 0: {
            return x;
        }
        case 1: {
            return y;
        }
        }

        REPORT_FATAL_ERROR();
        return x;
    }

    PBRT_CPU_GPU
    Real length_squared() const {
        return x * x + y * y;
    }
};

using Vector2f = Vector2<Real>;
using Vector2i = Vector2<int>;

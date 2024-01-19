#pragma once

#include <stdexcept>
#include <euclidean_space/point2.h>

template <typename T>
class Vector2 {
  public:
    T x, y;

    PBRT_CPU_GPU Vector2(T _x, T _y) : x(_x), y(_y) {}

    PBRT_CPU_GPU T &operator[](int index) {
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

using Vector2f = Vector2<double>;

PBRT_CPU_GPU Vector2f operator-(const Point2f &left, const Point2f &right) {
    return Vector2f(left.x - right.x, left.y - right.y);
}

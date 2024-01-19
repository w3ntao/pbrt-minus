#pragma once

#include "util/macro.h"

template <typename T>
class Point2 {
  public:
    T x, y;

    PBRT_CPU_GPU Point2() : x(NAN), y(NAN){};

    PBRT_CPU_GPU Point2(T _x, T _y) : x(_x), y(_y){};

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
            throw std::runtime_error("Point2: invalid index `" + std::to_string(index) + "`");
#endif
        }
        }
    }

    PBRT_CPU_GPU T operator[](int index) const {
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
            throw std::runtime_error("Point: invalid index `" + std::to_string(index) + "`");
#endif
        }
        }
    }

    PBRT_CPU_GPU Point2 operator+(const Point2 &right) const {
        return Point2(x + right.x, y + right.y);
    }

    PBRT_CPU_GPU Point2 operator*(T factor) const {
        return Point2(x * factor, y * factor);
    }
};

using Point2f = Point2<double>;

template <typename T>
PBRT_CPU_GPU Point2<T> operator*(T factor, const Point2<T> &p) {
    return p * factor;
}
#pragma once

#include <iostream>
#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/vector2.h"

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

    PBRT_CPU_GPU bool operator==(const Point2 &p) const {
        return x == p.x && y == p.y;
    }

    PBRT_CPU_GPU bool operator!=(const Point2 &p) const {
        return !((*this) == p);
    }

    PBRT_CPU_GPU Point2 operator+(const Point2 &p) const {
        return Point2(x + p.x, y + p.y);
    }

    PBRT_CPU_GPU Point2 operator+(const Vector2<T> &v) const {
        return Point2(x + v.x, y + v.y);
    }

    PBRT_CPU_GPU Vector2<T> operator-(const Point2<T> &right) {
        return Vector2<T>(x - right.x, y - right.y);
    }

    PBRT_CPU_GPU Point2 operator-(const Vector2f &v) const {
        return Point2(x - v.x, y - v.y);
    }

    PBRT_CPU_GPU Point2 operator*(T factor) const {
        return Point2(x * factor, y * factor);
    }

    PBRT_CPU_GPU Point2 min(const Point2 &p) const {
        return Point2(std::min(x, p.x), std::min(y, p.y));
    }

    PBRT_CPU_GPU Point2 max(const Point2 &p) const {
        return Point2(std::max(x, p.x), std::max(y, p.y));
    }

    friend std::ostream &operator<<(std::ostream &stream, const Point2 &p) {
        stream << "Point2(" << p.x << ", " << p.y << ")";
        return stream;
    }

    PBRT_CPU_GPU Point2<double> to_point2f() const {
        return Point2<double>(double(x), double(y));
    }
};

using Point2f = Point2<double>;
using Point2i = Point2<int>;

template <typename T>
PBRT_CPU_GPU Point2<T> operator*(T factor, const Point2<T> &p) {
    return p * factor;
}

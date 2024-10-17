#pragma once

#include "pbrt/euclidean_space/vector2.h"
#include "pbrt/util/macro.h"
#include <iostream>

template <typename T>
class Point2 {
  public:
    T x, y;

    PBRT_CPU_GPU Point2() {};

    PBRT_CPU_GPU Point2(T _x, T _y) : x(_x), y(_y) {};

    PBRT_CPU_GPU
    explicit Point2(const Vector2<T> v) : x(v.x), y(v.y) {}

    PBRT_CPU_GPU T &operator[](uint8_t index) {
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

    PBRT_CPU_GPU T operator[](uint8_t index) const {
        switch (index) {
        case 0: {
            return x;
        }
        case 1: {
            return y;
        }
        }

        REPORT_FATAL_ERROR();
        return NAN;
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

    PBRT_CPU_GPU void operator*=(T factor) {
        *this = *this * factor;
    }

    PBRT_CPU_GPU Point2 operator/(T divisor) const {
        return Point2(x / divisor, y / divisor);
    }

    PBRT_CPU_GPU void operator/=(T divisor) {
        *this = *this / divisor;
    }

    PBRT_CPU_GPU Point2 min(const Point2 &p) const {
        return Point2(std::min(x, p.x), std::min(y, p.y));
    }

    PBRT_CPU_GPU Point2 max(const Point2 &p) const {
        return Point2(std::max(x, p.x), std::max(y, p.y));
    }

    PBRT_CPU_GPU Point2<FloatType> to_point2f() const {
        return Point2<FloatType>(FloatType(x), FloatType(y));
    }

    PBRT_CPU_GPU Vector2<FloatType> to_vector2f() const {
        return Vector2<FloatType>(FloatType(x), FloatType(y));
    }

    friend std::ostream &operator<<(std::ostream &stream, const Point2 &p) {
        stream << "[ " << p.x << ", " << p.y << " ]";
        return stream;
    }
};

using Point2f = Point2<FloatType>;
using Point2i = Point2<int>;

template <typename T>
PBRT_CPU_GPU Point2<T> operator*(T factor, const Point2<T> &p) {
    return p * factor;
}

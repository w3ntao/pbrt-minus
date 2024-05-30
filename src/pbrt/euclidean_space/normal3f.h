#pragma once

#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/vector3.h"
#include "pbrt/util/utility_math.h"

class Normal3f {
  public:
    FloatType x;
    FloatType y;
    FloatType z;

    PBRT_CPU_GPU Normal3f() : x(NAN), y(NAN), z(NAN) {}

    PBRT_CPU_GPU Normal3f(FloatType _x, FloatType _y, FloatType _z) : x(_x), y(_y), z(_z) {}

    PBRT_CPU_GPU explicit Normal3f(const Vector3f &v) : x(v.x), y(v.y), z(v.z) {}

    PBRT_CPU_GPU
    bool has_nan() const {
        return std::isnan(x) || std::isnan(y) || std::isnan(z);
    }

    PBRT_CPU_GPU Vector3f to_vector3() const {
        return Vector3f(x, y, z);
    }

    PBRT_CPU_GPU
    FloatType dot(const Normal3f &n) const {
        return FMA(x, n.x, sum_of_products(y, n.y, z, n.z));
    }

    PBRT_CPU_GPU
    FloatType dot(const Vector3f &v) const {
        return FMA(x, v.x, sum_of_products(y, v.y, z, v.z));
    }

    PBRT_CPU_GPU
    FloatType abs_dot(const Normal3f &n) const {
        return std::abs(this->dot(n));
    }

    PBRT_CPU_GPU
    FloatType abs_dot(const Vector3f &v) const {
        return std::abs(this->dot(v));
    }

    PBRT_CPU_GPU Normal3f face_forward(const Normal3f &n) const {
        return this->dot(n) < 0.0 ? (-*this) : (*this);
    }

    PBRT_CPU_GPU bool operator==(const Normal3f &n) const {
        return x == n.x && y == n.y && z == n.z;
    }

    PBRT_CPU_GPU bool operator!=(const Normal3f &n) const {
        return !this->operator==(n);
    }

    PBRT_CPU_GPU Normal3f operator+(const Normal3f &n) const {
        return Normal3f(x + n.x, y + n.y, z + n.z);
    }

    PBRT_CPU_GPU Normal3f operator-() const {
        return Normal3f(-x, -y, -z);
    }

    PBRT_CPU_GPU Normal3f operator*(FloatType factor) const {
        return Normal3f(x * factor, y * factor, z * factor);
    }

    PBRT_CPU_GPU void operator*=(FloatType factor) {
        *this = *this * factor;
    }

    friend std::ostream &operator<<(std::ostream &stream, const Normal3f &n) {
        stream << "Normal3(" << n.x << ", " << n.y << ", " << n.z << ")";
        return stream;
    }
};

PBRT_CPU_GPU
static Normal3f operator*(FloatType factor, const Normal3f &n) {
    return n * factor;
}
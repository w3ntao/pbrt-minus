#pragma once

#include <pbrt/euclidean_space/vector3.h>
#include <pbrt/gpu/macro.h>
#include <pbrt/util/math.h>

class Normal3f {
  public:
    FloatType x;
    FloatType y;
    FloatType z;

    PBRT_CPU_GPU
    Normal3f() : x(NAN), y(NAN), z(NAN) {}

    PBRT_CPU_GPU
    Normal3f(FloatType _x, FloatType _y, FloatType _z) : x(_x), y(_y), z(_z) {}

    PBRT_CPU_GPU
    explicit Normal3f(const Vector3f &v) : x(v.x), y(v.y), z(v.z) {}

    PBRT_CPU_GPU
    FloatType operator[](uint8_t index) const {
        switch (index) {
        case 0: {
            return x;
        }
        case 1: {
            return y;
        }
        case 2: {
            return z;
        }
        }

        REPORT_FATAL_ERROR();
        return NAN;
    }

    PBRT_CPU_GPU
    bool has_nan() const {
        return isnan(x) || isnan(y) || isnan(z) || isinf(x) || isinf(y) || isinf(z);
    }

    PBRT_CPU_GPU
    Vector3f to_vector3() const {
        return Vector3f(x, y, z);
    }

    PBRT_CPU_GPU
    Normal3f abs() const {
        return Normal3f(std::abs(x), std::abs(y), std::abs(z));
    }

    PBRT_CPU_GPU
    FloatType squared_length() const {
        return sqr(x) + sqr(y) + sqr(z);
    }

    PBRT_CPU_GPU
    FloatType length() const {
        return std::sqrt(sqr(x) + sqr(y) + sqr(z));
    }

    PBRT_CPU_GPU
    Normal3f normalize() const {
        return *this / length();
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

    PBRT_CPU_GPU
    Vector3f cross(const Normal3f &right) const {
        return Vector3(difference_of_products(y, right.z, z, right.y),
                       difference_of_products(z, right.x, x, right.z),
                       difference_of_products(x, right.y, y, right.x));
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

    PBRT_CPU_GPU
    Normal3f operator+(const Normal3f &n) const {
        return Normal3f(x + n.x, y + n.y, z + n.z);
    }

    PBRT_CPU_GPU
    Normal3f operator-() const {
        return Normal3f(-x, -y, -z);
    }

    PBRT_CPU_GPU
    Normal3f operator-(const Normal3f &right) const {
        return Normal3f(x - right.x, y - right.y, z - right.z);
    }

    PBRT_CPU_GPU Normal3f operator*(FloatType factor) const {
        return Normal3f(x * factor, y * factor, z * factor);
    }

    PBRT_CPU_GPU void operator*=(FloatType factor) {
        *this = *this * factor;
    }

    PBRT_CPU_GPU Normal3f operator/(FloatType divisor) const {
        return Normal3f(x / divisor, y / divisor, z / divisor);
    }

    friend std::ostream &operator<<(std::ostream &stream, const Normal3f &n) {
        stream << std::setprecision(4) << "[" << n.x << ", " << n.y << ", " << n.z << "]";
        return stream;
    }
};

PBRT_CPU_GPU
static Normal3f operator*(const FloatType factor, const Normal3f &n) {
    return n * factor;
}

PBRT_CPU_GPU
static inline Normal3f FMA(const FloatType a, const Normal3f &b, const Normal3f &c) {
    return {FMA(a, b.x, c.x), FMA(a, b.y, c.y), FMA(a, b.z, c.z)};
}

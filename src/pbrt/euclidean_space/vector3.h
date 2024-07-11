#pragma once

#include <stdexcept>

#include "pbrt/util/utility_math.h"
#include "pbrt/util/interval.h"

template <typename T>
class Vector3 {
  public:
    T x, y, z;

    PBRT_CPU_GPU Vector3() : x(NAN), y(NAN), z(NAN) {};

    PBRT_CPU_GPU Vector3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}

    PBRT_CPU_GPU T &operator[](uint8_t index) {
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
        case 2: {
            return z;
        }
        }

        REPORT_FATAL_ERROR();
        return NAN;
    }

    PBRT_CPU_GPU bool operator==(const Vector3 &v) const {
        return x == v.x && y == v.y && z == v.z;
    }

    PBRT_CPU_GPU Vector3 operator+(const Vector3 &b) const {
        return Vector3(x + b.x, y + b.y, z + b.z);
    }

    PBRT_CPU_GPU void operator+=(const Vector3 &v) {
        this->x += v.x;
        this->y += v.y;
        this->z += v.z;
    }

    PBRT_CPU_GPU Vector3 operator-() const {
        return Vector3(-x, -y, -z);
    }

    PBRT_CPU_GPU Vector3 operator-(const Vector3 &b) const {
        return Vector3(x - b.x, y - b.y, z - b.z);
    }

    PBRT_CPU_GPU Vector3 operator*(T factor) const {
        return Vector3(x * factor, y * factor, z * factor);
    }

    PBRT_CPU_GPU friend Vector3 operator*(T factor, const Vector3 &v) {
        return v * factor;
    }

    PBRT_CPU_GPU Vector3 operator*=(FloatType v) {
        this->x += v;
        this->y += v;
        this->z += v;

        return *this;
    }

    PBRT_CPU_GPU Vector3 operator/(T divisor) const {
        return Vector3(x / divisor, y / divisor, z / divisor);
    }

    PBRT_CPU_GPU uint max_component_index() const {
        return (x > y) ? ((x > z) ? 0 : 2) : ((y > z) ? 1 : 2);
    }

    PBRT_CPU_GPU T max_component_value() const {
        return std::max({x, y, z});
    }

    PBRT_CPU_GPU Vector3 permute(const uint8_t indices[3]) const {
        T _x = this->operator[](indices[0]);
        T _y = this->operator[](indices[1]);
        T _z = this->operator[](indices[2]);

        return Vector3(_x, _y, _z);
    }

    PBRT_CPU_GPU Vector3 abs() const {
        return Vector3(std::abs(x), std::abs(y), std::abs(z));
    }

    PBRT_CPU_GPU T squared_length() const {
        return x * x + y * y + z * z;
    }

    PBRT_CPU_GPU T length() const {
        return std::sqrt(squared_length());
    }

    PBRT_CPU_GPU Vector3 normalize() const {
        return *this / length();
    }

    PBRT_GPU Vector3 softmax() const {
        auto v = Vector3(exp10f(x), exp10f(y), exp10f(z));
        return v / (v.x + v.y + v.z);
    }

    PBRT_CPU_GPU
    inline T dot(const Vector3 &right) const {
        return FMA(x, right.x, sum_of_products(y, right.y, z, right.z));
    }

    PBRT_CPU_GPU
    inline T abs_dot(const Vector3 &right) const {
        return std::abs(this->dot(right));
    }

    PBRT_CPU_GPU Vector3 cross(const Vector3 &right) const {
        // TODO: rewrite this with difference_of_products()
        return Vector3(y * right.z - z * right.y, -(x * right.z - z * right.x),
                       x * right.y - y * right.x);
    }

    PBRT_CPU_GPU Vector3 face_forward(const Vector3 &v) const {
        return this->dot(v) < 0.0 ? (-*this) : (*this);
    }

    PBRT_CPU_GPU void coordinate_system(Vector3 *v2, Vector3 *v3) const {
        FloatType sign = std::copysign(1.0, z);
        FloatType a = -1.0 / (sign + z);
        FloatType b = x * y * a;
        *v2 = Vector3(1 + sign * sqr(x) * a, sign * b, -sign * x);
        *v3 = Vector3(b, sign + sqr(y) * a, -y);
    }

    PBRT_CPU_GPU
    bool same_hemisphere(const Vector3 &wp) const {
        return z * wp.z > 0;
    }

    PBRT_CPU_GPU FloatType abs_cos_theta() const {
        return std::abs(z);
    }

    PBRT_CPU_GPU inline FloatType cos_theta() const {
        return z;
    }

    PBRT_CPU_GPU inline FloatType cos2_theta() const {
        return sqr(z);
    }

    PBRT_CPU_GPU inline FloatType sin2_theta() const {
        return std::max<FloatType>(0.0, 1.0 - cos2_theta());
    }

    PBRT_CPU_GPU inline FloatType sin_theta() const {
        return std::sqrt(sin2_theta());
    }

    PBRT_CPU_GPU inline FloatType tan_theta() const {
        return sin_theta() / cos_theta();
    }

    PBRT_CPU_GPU inline FloatType tan2_theta() const {
        return sin2_theta() / cos2_theta();
    }

    PBRT_CPU_GPU inline FloatType cos_phi() const {
        FloatType sinTheta = sin_theta();
        return (sinTheta == 0) ? 1 : clamp<FloatType>(x / sinTheta, -1, 1);
    }

    PBRT_CPU_GPU inline FloatType sin_phi() const {
        FloatType sinTheta = sin_theta();
        return (sinTheta == 0) ? 0 : clamp<FloatType>(y / sinTheta, -1, 1);
    }

    PBRT_CPU_GPU void print() const {
        printf("(%f, %f, %f)", x, y, z);
    }
};

using Vector3f = Vector3<FloatType>;

PBRT_CPU_GPU inline Vector3f FMA(FloatType a, const Vector3f &b, const Vector3f &c) {
    return {FMA(a, b.x, c.x), FMA(a, b.y, c.y), FMA(a, b.z, c.z)};
}

PBRT_CPU_GPU inline Vector3f FMA(const Vector3f &a, FloatType b, const Vector3f &c) {
    return FMA(b, a, c);
}

// Equivalent to std::acos(Dot(a, b)), but more numerically stable.
// via http://www.plunk.org/~hatch/rightway.html
template <typename T>
PBRT_CPU_GPU inline FloatType angle_between(Vector3<T> v1, Vector3<T> v2) {
    if (v1.dot(v2) < 0) {
        return compute_pi() - 2 * safe_asin((v1 + v2).length() / 2);
    } else {
        return 2 * safe_asin((v2 - v1).length() / 2);
    }
}

template <typename T>
PBRT_CPU_GPU inline Vector3<T> gram_schmidt(Vector3<T> v, Vector3<T> w) {
    return v - v.dot(w) * w;
}

PBRT_CPU_GPU
static Vector3f SphericalDirection(FloatType sinTheta, FloatType cosTheta, FloatType phi) {
    return Vector3f(clamp<FloatType>(sinTheta, -1, 1) * std::cos(phi),
                    clamp<FloatType>(sinTheta, -1, 1) * std::sin(phi),
                    clamp<FloatType>(cosTheta, -1, 1));
}

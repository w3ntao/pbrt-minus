#pragma once

#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/vector3.h"
#include "pbrt/util/math.h"

class Normal3f {
  public:
    double x;
    double y;
    double z;

    PBRT_CPU_GPU Normal3f() : x(NAN), y(NAN), z(NAN) {}

    PBRT_CPU_GPU Normal3f(double _x, double _y, double _z) : x(_x), y(_y), z(_z) {}

    PBRT_CPU_GPU explicit Normal3f(const Vector3f &v) : x(v.x), y(v.y), z(v.z) {}

    PBRT_CPU_GPU Vector3f to_vector3() const {
        return Vector3f(x, y, z);
    }

    PBRT_CPU_GPU bool operator==(const Normal3f &n) const {
        return x == n.x && y == n.y && z == n.z;
    }

    PBRT_CPU_GPU bool operator!=(const Normal3f &n) const {
        return !this->operator==(n);
    }

    PBRT_CPU_GPU Normal3f operator-() const {
        return Normal3f(-x, -y, -z);
    }

    PBRT_CPU_GPU Normal3f operator*=(double factor) {
        x *= factor;
        y *= factor;
        z *= factor;
        return *this;
    }

    PBRT_CPU_GPU Normal3f abs() const {
        return Normal3f(std::abs(x), std::abs(y), std::abs(z));
    }

    PBRT_CPU_GPU double dot(const Vector3f &v) const {
        return FMA(x, v.x, sum_of_products(y, v.y, z, v.z));
    }
};

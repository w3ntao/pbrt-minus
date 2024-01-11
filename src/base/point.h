#pragma once

#include <stdexcept>
#include "vector3.h"

class Point {
    public:
        double x, y, z;

        PBRT_CPU_GPU Point() : x(0.0), y(0.0), z(0.0){};

        PBRT_CPU_GPU Point(double _x, double _y, double _z) : x(_x), y(_y), z(_z){};

        PBRT_CPU_GPU double &operator[](int index) {
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
            default: {
#if defined(__CUDA_ARCH__)
                asm("trap;");
#else
                throw std::runtime_error("Point: invalid index `" + std::to_string(index) + "`");
#endif
            }
            }
        }

        PBRT_CPU_GPU double operator[](int index) const {
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
            default: {
#if defined(__CUDA_ARCH__)
                asm("trap;");
#else
                throw std::runtime_error("Point: invalid index `" + std::to_string(index) + "`");
#endif
            }
            }
        }

        PBRT_GPU inline Vector3 to_vector() const {
            return Vector3(x, y, z);
        }
};

PBRT_GPU inline Point operator+(const Point &p, const Vector3 &v) {
    return Point(p.x + v.x, p.y + v.y, p.z + v.z);
}

PBRT_GPU inline Vector3 operator-(const Point &left, const Point &right) {
    return Vector3(left.x - right.x, left.y - right.y, left.z - right.z);
}

PBRT_GPU inline Point operator-(const Point &p, const Vector3 &v) {
    return Point(p.x - v.x, p.y - v.y, p.z - v.z);
}

PBRT_GPU inline Vector3 operator-(const Vector3 &v, const Point &p) {
    return Vector3(v.x - p.x, v.y - p.y, v.z - p.z);
}

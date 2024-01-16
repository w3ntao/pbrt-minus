#pragma once

#include <stdexcept>
#include "vector3.h"

template <typename T>
class Point3 {
    public:
        T x, y, z;

        PBRT_CPU_GPU Point3() : x(0.0), y(0.0), z(0.0){};

        PBRT_CPU_GPU Point3(T _x, T _y, T _z) : x(_x), y(_y), z(_z){};

        PBRT_CPU_GPU T &operator[](int index) {
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

        PBRT_CPU_GPU T operator[](int index) const {
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

        PBRT_CPU_GPU Point3<T> operator+(const Vector3<T> &v) const {
            return Point3<T>(x + v.x, y + v.y, z + v.z);
        }

        PBRT_CPU_GPU Vector3<T> operator-(const Point3 &right) const {
            return Vector3<T>(x - right.x, y - right.y, z - right.z);
        }

        PBRT_CPU_GPU Point3<T> operator-(const Vector3<T> &v) const {
            return Point3(x - v.x, y - v.y, z - v.z);
        }

        PBRT_CPU_GPU Point3<T> operator/(T divisor) const {
            return Point3(x / divisor, y / divisor, z / divisor);
        }

        PBRT_CPU_GPU Vector3<T> to_vector() const {
            return Vector3<T>(x, y, z);
        }
};

using Point3f = Point3<double>;

template <typename T>
PBRT_CPU_GPU Vector3<T> operator-(const Vector3<T> &v, const Point3<T> &p) {
    return Vector3<T>(v.x - p.x, v.y - p.y, v.z - p.z);
}

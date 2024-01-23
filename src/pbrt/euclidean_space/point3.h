#pragma once

#include <stdexcept>

#include "pbrt/euclidean_space/vector3.h"

template <typename T>
class Point3 {
  public:
    T x, y, z;

    PBRT_CPU_GPU Point3() : x(NAN), y(NAN), z(NAN){};

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

    PBRT_CPU_GPU Point3 operator+(const Point3 &right) const {
        return Point3(x + right.x, y + right.y, z + right.z);
    }

    PBRT_CPU_GPU Point3 operator*(T factor) const {
        return Point3(x * factor, y * factor, z * factor);
    }

    PBRT_CPU_GPU Point3 operator*=(T factor) {
        x *= factor;
        y *= factor;
        z *= factor;
        return *this;
    }

    PBRT_CPU_GPU Point3 operator/(T divisor) const {
        return Point3(x / divisor, y / divisor, z / divisor);
    }

    PBRT_CPU_GPU Vector3<T> to_vector3() const {
        return Vector3<T>(x, y, z);
    }

    PBRT_CPU_GPU Point3 abs() const {
        return Point3(std::abs(x), std::abs(y), std::abs(z));
    }

    PBRT_CPU_GPU Point3 permute(const std::array<int, 3> &p) const {
        T _x = this->operator[](p[0]);
        T _y = this->operator[](p[1]);
        T _z = this->operator[](p[2]);

        return Point3(_x, _y, _z);
    }
};

using Point3f = Point3<double>;

template <typename T>
PBRT_CPU_GPU Point3<T> operator*(T factor, const Point3<T> &p) {
    return p * factor;
}

template <typename T>
PBRT_CPU_GPU Vector3<T> operator-(const Vector3<T> &v, const Point3<T> &p) {
    return Vector3<T>(v.x - p.x, v.y - p.y, v.z - p.z);
}

template <typename T>
PBRT_CPU_GPU Point3<T> operator+(const Point3<T> &p, const Vector3<T> &v) {
    return Point3<T>(p.x + v.x, p.y + v.y, p.z + v.z);
}

template <typename T>
PBRT_CPU_GPU Vector3<T> operator-(const Point3<T> &left, const Point3<T> &right) {
    return Vector3<T>(left.x - right.x, left.y - right.y, left.z - right.z);
}

template <typename T>
PBRT_CPU_GPU Point3<T> operator-(const Point3<T> &p, const Vector3<T> &v) {
    return Point3(p.x - v.x, p.y - v.y, p.z - v.z);
}

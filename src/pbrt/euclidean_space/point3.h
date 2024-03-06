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
            printf("Point3: invalid index `%d`\n\n", index);
#if defined(__CUDA_ARCH__)
            asm("trap;");
#else
            throw std::runtime_error("Point: invalid index\n");
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

    PBRT_CPU_GPU bool operator==(const Point3 &p) const {
        return x == p.x && y == p.y && z == p.z;
    }

    PBRT_CPU_GPU Point3 operator+(const Point3 &p) const {
        return Point3(x + p.x, y + p.y, z + p.z);
    }

    PBRT_CPU_GPU Point3 operator+(const Vector3<T> &v) const {
        return Point3(x + v.x, y + v.y, z + v.z);
    }

    PBRT_CPU_GPU Vector3<T> operator-(const Point3 &p) const {
        return Vector3<T>(x - p.x, y - p.y, z - p.z);
    }

    PBRT_CPU_GPU Point3 operator-(const Vector3<T> &v) const {
        return Point3(x - v.x, y - v.y, z - v.z);
    }

    PBRT_CPU_GPU friend Vector3<T> operator-(const Vector3<T> &v, const Point3 &p) {
        return Vector3<T>(v.x - p.x, v.y - p.y, v.z - p.z);
    }

    PBRT_CPU_GPU Point3 operator*(T factor) const {
        return Point3(x * factor, y * factor, z * factor);
    }

    PBRT_CPU_GPU friend Point3 operator*(T factor, const Point3 &p) {
        return p * factor;
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

    PBRT_CPU_GPU Point3 min(const Point3 &p) const {
        return Point3(std::min(x, p.x), std::min(y, p.y), std::min(z, p.z));
    }

    PBRT_CPU_GPU Point3 max(const Point3 &p) const {
        return Point3(std::max(x, p.x), std::max(y, p.y), std::max(z, p.z));
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

    friend std::ostream &operator<<(std::ostream &stream, const Point3 &p) {
        stream << "Point3(" << p.x << ", " << p.y << ", " << p.z << ")";
        return stream;
    }
};

using Point3f = Point3<double>;

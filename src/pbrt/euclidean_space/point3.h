#pragma once

#include "pbrt/euclidean_space/vector3.h"

template <typename T>
class Point3 {
  public:
    T x, y, z;

    PBRT_CPU_GPU Point3() : x(NAN), y(NAN), z(NAN){};

    PBRT_CPU_GPU Point3(T _x, T _y, T _z) : x(_x), y(_y), z(_z){};

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

    PBRT_CPU_GPU bool operator==(const Point3 &p) const {
        return x == p.x && y == p.y && z == p.z;
    }

    PBRT_CPU_GPU Point3<T> operator+(const Point3<T> &p) const {
        return Point3<T>(x + p.x, y + p.y, z + p.z);
    }

    PBRT_CPU_GPU Point3<T> operator+(const Vector3<T> &v) const {
        return Point3<T>(x + v.x, y + v.y, z + v.z);
    }

    PBRT_CPU_GPU void operator+=(const Point3<T> &p) {
        for (int idx = 0; idx < 3; idx++) {
            (*this)[idx] += p[idx];
        }
    }

    PBRT_CPU_GPU
    Point3 operator-() const {
        return Point3(-x, -y, -z);
    }

    PBRT_CPU_GPU Vector3<T> operator-(const Point3<T> &p) const {
        return Vector3<T>(x - p.x, y - p.y, z - p.z);
    }

    PBRT_CPU_GPU Point3<T> operator-(const Vector3<T> &v) const {
        return Point3<T>(x - v.x, y - v.y, z - v.z);
    }

    PBRT_CPU_GPU friend Vector3<T> operator-(const Vector3<T> &v, const Point3<T> &p) {
        return Vector3<T>(v.x - p.x, v.y - p.y, v.z - p.z);
    }

    PBRT_CPU_GPU Point3<T> operator*(T factor) const {
        return Point3<T>(x * factor, y * factor, z * factor);
    }

    PBRT_CPU_GPU friend Point3<T> operator*(T factor, const Point3<T> &p) {
        return p * factor;
    }

    PBRT_CPU_GPU Point3<T> operator*=(T factor) {
        x *= factor;
        y *= factor;
        z *= factor;
        return *this;
    }

    PBRT_CPU_GPU Point3<T> operator/(T divisor) const {
        return Point3<T>(x / divisor, y / divisor, z / divisor);
    }

    PBRT_CPU_GPU
    void operator/=(T divisor) {
        *this = *this / divisor;
    }

    PBRT_CPU_GPU Point3<T> min(const Point3<T> &p) const {
        return Point3<T>(std::min(x, p.x), std::min(y, p.y), std::min(z, p.z));
    }

    PBRT_CPU_GPU Point3<T> max(const Point3<T> &p) const {
        return Point3<T>(std::max(x, p.x), std::max(y, p.y), std::max(z, p.z));
    }

    PBRT_CPU_GPU
    bool has_nan() const {
        return isnan(x) || isnan(y) || isnan(z) || isinf(x) || isinf(y) || isinf(z);
    }

    PBRT_CPU_GPU Vector3<T> to_vector3() const {
        return Vector3<T>(x, y, z);
    }

    PBRT_CPU_GPU Point3<T> abs() const {
        return Point3<T>(std::abs(x), std::abs(y), std::abs(z));
    }

    PBRT_CPU_GPU Point3<T> permute(const uint8_t indices[3]) const {
        T _x = this->operator[](indices[0]);
        T _y = this->operator[](indices[1]);
        T _z = this->operator[](indices[2]);

        return Point3<T>(_x, _y, _z);
    }

    PBRT_CPU_GPU
    FloatType squared_distance(const Point3<T> &p) const {
        return sqr(x - p.x) + sqr(y - p.y) + sqr(z - p.z);
    }

    PBRT_CPU_GPU
    FloatType distance(const Point3<T> &p) const {
        return std::sqrt(this->squared_distance(p));
    }

    friend std::ostream &operator<<(std::ostream &stream, const Point3 &p) {
        stream << "Point3(" << p.x << ", " << p.y << ", " << p.z << ")";
        return stream;
    }
};

using Point3f = Point3<FloatType>;

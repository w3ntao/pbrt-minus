#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/squared_matrix.h"
#include "pbrt/util/macro.h"

class XYZ {
  public:
    PBRT_CPU_GPU
    XYZ(FloatType _x, FloatType _y, FloatType _z) : x(_x), y(_y), z(_z) {}

    PBRT_CPU_GPU
    static XYZ from_xyY(const Point2f &xy, FloatType Y = 1) {
        if (xy.y == 0) {
            return {0, 0, 0};
        }

        return {xy.x * Y / xy.y, Y, (1 - xy.x - xy.y) * Y / xy.y};
    }

    PBRT_CPU_GPU
    bool operator==(const XYZ &right) const {
        return x == right.x && y == right.y && z == right.z;
    }

    PBRT_CPU_GPU
    bool operator!=(const XYZ &right) const {
        return x != right.x || y != right.y || z != right.z;
    }

    PBRT_CPU_GPU
    FloatType operator[](uint8_t idx) const {
        switch (idx) {
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
    FloatType &operator[](uint8_t idx) {
        switch (idx) {
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

    PBRT_CPU_GPU
    XYZ operator+(const XYZ &right) const {
        return {x + right.x, y + right.y, z + right.z};
    }

    PBRT_CPU_GPU
    void operator+=(const XYZ &right) {
        *this = *this + right;
    }

    PBRT_CPU_GPU
    XYZ operator-(const XYZ &right) const {
        return {x - right.x, y - right.y, z - right.z};
    }

    PBRT_CPU_GPU
    void operator-=(const XYZ &right) {
        *this = *this - right;
    }

    PBRT_CPU_GPU
    XYZ operator*(const XYZ &right) const {
        return {x * right.x, y * right.y, z * right.z};
    }

    PBRT_CPU_GPU
    XYZ operator*(FloatType a) const {
        return {a * x, a * y, a * z};
    }

    PBRT_CPU_GPU
    friend XYZ operator*(FloatType a, const XYZ &s) {
        return s * a;
    }

    PBRT_CPU_GPU
    friend XYZ operator*(const SquareMatrix<3> &m, const XYZ &rhs) {
        return XYZ(inner_product(m[0][0], rhs.x, m[0][1], rhs.y, m[0][2], rhs.z),
                   inner_product(m[1][0], rhs.x, m[1][1], rhs.y, m[1][2], rhs.z),
                   inner_product(m[2][0], rhs.x, m[2][1], rhs.y, m[2][2], rhs.z));
    }

    PBRT_CPU_GPU
    void operator*=(const XYZ &s) {
        *this = *this * s;
    }

    PBRT_CPU_GPU
    void operator*=(FloatType a) {
        *this = *this * a;
    }

    PBRT_CPU_GPU
    XYZ operator/(const XYZ &right) const {
        return {x / right.x, y / right.y, z / right.z};
    }

    PBRT_CPU_GPU
    XYZ operator/(FloatType a) const {
        return {x / a, y / a, z / a};
    }

    PBRT_CPU_GPU
    void operator/=(const XYZ &right) {
        *this = *this / right;
    }

    PBRT_CPU_GPU
    void operator/=(FloatType a) {
        *this = *this / a;
    }

    PBRT_CPU_GPU
    XYZ operator-() const {
        return {-x, -y, -z};
    }

    PBRT_CPU_GPU
    Point2f xy() const {
        return {x / (x + y + z), y / (x + y + z)};
    }

    // XYZ Public Members
    FloatType x;
    FloatType y;
    FloatType z;
};

#pragma once

#include "pbrt/euclidean_space/squared_matrix.h"
#include "pbrt/util/basic_math.h"

class RGB {
  public:
    FloatType r, g, b;

    PBRT_CPU_GPU RGB() : r(0.0), g(0.0), b(0.0) {}

    PBRT_CPU_GPU RGB(FloatType x) : r(x), g(x), b(x) {}

    PBRT_CPU_GPU RGB(FloatType _r, FloatType _g, FloatType _b) : r(_r), g(_g), b(_b) {}

    PBRT_CPU_GPU FloatType sum() const {
        return r + g + b;
    }

    PBRT_CPU_GPU FloatType avg() const {
        return sum() / 3;
    }

    PBRT_CPU_GPU
    RGB clamp(FloatType low, FloatType high) const {
        return RGB(::clamp(r, low, high), ::clamp(g, low, high), ::clamp(b, low, high));
    }

    PBRT_CPU_GPU
    bool has_nan() const {
        return isnan(r) || isnan(g) || isnan(b) || isinf(r) || isinf(g) || isinf(b);
    }

    PBRT_CPU_GPU FloatType max_component() const {
        return std::max(std::max(r, g), b);
    }

    PBRT_CPU_GPU
    FloatType operator[](uint8_t idx) const {
        switch (idx) {
        case 0: {
            return r;
        }
        case 1: {
            return g;
        }
        case 2: {
            return b;
        }
        }

        REPORT_FATAL_ERROR();
        return NAN;
    }

    PBRT_CPU_GPU
    FloatType &operator[](uint8_t index) {
        switch (index) {
        case 0: {
            return r;
        }
        case 1: {
            return g;
        }
        case 2: {
            return b;
        }
        }

        REPORT_FATAL_ERROR();
        return r;
    }

    PBRT_CPU_GPU bool operator==(const RGB &rhs) const {
        return r == rhs.r && g == rhs.g && b == rhs.b;
    }

    PBRT_CPU_GPU bool operator!=(const RGB &rhs) const {
        return !(*this == rhs);
    }

    PBRT_CPU_GPU RGB operator+(const RGB &right) const {
        return RGB(r + right.r, g + right.g, b + right.b);
    }

    PBRT_CPU_GPU RGB operator-(const RGB &right) const {
        return RGB(r - right.r, g - right.g, b - right.b);
    }

    PBRT_CPU_GPU RGB operator*(const RGB &right) const {
        return RGB(r * right.r, g * right.g, b * right.b);
    }

    PBRT_CPU_GPU RGB operator*(FloatType scalar) const {
        return RGB(r * scalar, g * scalar, b * scalar);
    }

    PBRT_CPU_GPU
    friend RGB operator*(const SquareMatrix<3> &m, const RGB &rgb) {
        return RGB(inner_product(m[0][0], rgb.r, m[0][1], rgb.g, m[0][2], rgb.b),
                   inner_product(m[1][0], rgb.r, m[1][1], rgb.g, m[1][2], rgb.b),
                   inner_product(m[2][0], rgb.r, m[2][1], rgb.g, m[2][2], rgb.b));
    }

    PBRT_CPU_GPU RGB operator/(FloatType divisor) const {
        return RGB(r / divisor, g / divisor, b / divisor);
    }

    PBRT_CPU_GPU void operator+=(const RGB &c) {
        r += c.r;
        g += c.g;
        b += c.b;
    }

    PBRT_CPU_GPU void operator*=(const RGB &c) {
        r *= c.r;
        g *= c.g;
        b *= c.b;
    }

    PBRT_CPU_GPU void operator/=(FloatType divisor) {
        r /= divisor;
        g /= divisor;
        b /= divisor;
    }

    PBRT_CPU_GPU
    friend RGB operator*(FloatType scalar, const RGB &c) {
        return RGB(c.r * scalar, c.g * scalar, c.b * scalar);
    }

    friend std::ostream &operator<<(std::ostream &stream, const RGB &rgb) {
        stream << "RGB(" << rgb.r << ", " << rgb.g << ", " << rgb.b << ")";
        return stream;
    }
};

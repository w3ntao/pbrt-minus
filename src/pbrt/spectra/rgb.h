#pragma once

#include "pbrt/util/math.h"

class RGB {
  public:
    double r, g, b;

    PBRT_CPU_GPU RGB() : r(0.0), g(0.0), b(0.0) {}

    PBRT_CPU_GPU RGB(double x) : r(x), g(x), b(x) {}

    PBRT_CPU_GPU RGB(double _r, double _g, double _b) : r(_r), g(_g), b(_b) {}

    PBRT_CPU_GPU RGB clamp_zero() const {
        return RGB(std::max<double>(0, r), std::max<double>(0, g), std::max<double>(0, b));
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

    PBRT_CPU_GPU RGB operator*(double scalar) const {
        return RGB(r * scalar, g * scalar, b * scalar);
    }

    PBRT_CPU_GPU RGB operator/(double divisor) const {
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

    PBRT_CPU_GPU void operator/=(double divisor) {
        r /= divisor;
        g /= divisor;
        b /= divisor;
    }

    PBRT_CPU_GPU double operator[](int idx) const {
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
        default: {
            printf("RGB: invalid index: `%d`\n", idx);

#if defined(__CUDA_ARCH__)
            asm("trap;");
#else
            throw std::runtime_error("RGB: invalid index\n");
#endif
        }
        }
    }
};

PBRT_CPU_GPU inline RGB operator*(double scalar, const RGB &c) {
    return RGB(c.r * scalar, c.g * scalar, c.b * scalar);
}

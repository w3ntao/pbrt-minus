#pragma once

#include "util/math.h"

class Color {
  public:
    double r, g, b;

    PBRT_CPU_GPU Color() : r(0.0), g(0.0), b(0.0) {}

    PBRT_CPU_GPU Color(double r, double g, double b) : r(r), g(g), b(b) {}

    PBRT_CPU_GPU Color clamp() const {
        return Color(gpu_clamp_0_1(r), gpu_clamp_0_1(g), gpu_clamp_0_1(b));
    }

    PBRT_GPU inline void operator+=(const Color &c) {
        r += c.r;
        g += c.g;
        b += c.b;
    }

    PBRT_GPU inline void operator*=(const Color &c) {
        r *= c.r;
        g *= c.g;
        b *= c.b;
    }

    PBRT_GPU inline void operator/=(double divisor) {
        r /= divisor;
        g /= divisor;
        b /= divisor;
    }
};

PBRT_GPU inline Color operator+(const Color &left, const Color &right) {
    return Color(left.r + right.r, left.g + right.g, left.b + right.b);
}

PBRT_GPU inline Color operator-(const Color &left, const Color &right) {
    return Color(left.r - right.r, left.g - right.g, left.b - right.b);
}

PBRT_GPU inline Color operator*(const Color &left, const Color &right) {
    return Color(left.r * right.r, left.g * right.g, left.b * right.b);
}

PBRT_GPU inline Color operator*(const Color &c, double scalar) {
    return Color(c.r * scalar, c.g * scalar, c.b * scalar);
}

PBRT_GPU inline Color operator*(double scalar, const Color &c) {
    return Color(c.r * scalar, c.g * scalar, c.b * scalar);
}

PBRT_GPU inline Color operator/(const Color c, double divisor) {
    return Color(c.r / divisor, c.g / divisor, c.b / divisor);
}

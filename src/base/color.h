#pragma once

#include "util/math.h"

class Color {
  public:
    double r, g, b;

    PBRT_CPU_GPU Color() : r(0.0), g(0.0), b(0.0) {}

    PBRT_CPU_GPU Color(double _r, double _g, double _b) : r(_r), g(_g), b(_b) {}

    PBRT_CPU_GPU Color clamp() const {
        return Color(gpu_clamp_0_1(r), gpu_clamp_0_1(g), gpu_clamp_0_1(b));
    }

    PBRT_GPU Color operator+(const Color &right) const {
        return Color(r + right.r, g + right.g, b + right.b);
    }

    PBRT_GPU Color operator-(const Color &right) const {
        return Color(r - right.r, g - right.g, b - right.b);
    }

    PBRT_GPU Color operator*(const Color &right) const {
        return Color(r * right.r, g * right.g, b * right.b);
    }

    PBRT_GPU Color operator*(double scalar) const {
        return Color(r * scalar, g * scalar, b * scalar);
    }

    PBRT_GPU Color operator/(double divisor) const {
        return Color(r / divisor, g / divisor, b / divisor);
    }

    PBRT_GPU void operator+=(const Color &c) {
        r += c.r;
        g += c.g;
        b += c.b;
    }

    PBRT_GPU void operator*=(const Color &c) {
        r *= c.r;
        g *= c.g;
        b *= c.b;
    }

    PBRT_GPU void operator/=(double divisor) {
        r /= divisor;
        g /= divisor;
        b /= divisor;
    }
};

PBRT_GPU inline Color operator*(double scalar, const Color &c) {
    return Color(c.r * scalar, c.g * scalar, c.b * scalar);
}

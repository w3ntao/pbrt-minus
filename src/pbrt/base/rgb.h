#pragma once

#include "pbrt/util/math.h"

class RGB {
  public:
    double r, g, b;

    PBRT_CPU_GPU RGB() : r(0.0), g(0.0), b(0.0) {}

    PBRT_CPU_GPU RGB(double x) : r(x), g(x), b(x) {}

    PBRT_CPU_GPU RGB(double _r, double _g, double _b) : r(_r), g(_g), b(_b) {}

    PBRT_CPU_GPU RGB clamp() const {
        return RGB(clamp_0_1(r), clamp_0_1(g), clamp_0_1(b));
    }

    PBRT_GPU RGB operator+(const RGB &right) const {
        return RGB(r + right.r, g + right.g, b + right.b);
    }

    PBRT_GPU RGB operator-(const RGB &right) const {
        return RGB(r - right.r, g - right.g, b - right.b);
    }

    PBRT_GPU RGB operator*(const RGB &right) const {
        return RGB(r * right.r, g * right.g, b * right.b);
    }

    PBRT_GPU RGB operator*(double scalar) const {
        return RGB(r * scalar, g * scalar, b * scalar);
    }

    PBRT_GPU RGB operator/(double divisor) const {
        return RGB(r / divisor, g / divisor, b / divisor);
    }

    PBRT_GPU void operator+=(const RGB &c) {
        r += c.r;
        g += c.g;
        b += c.b;
    }

    PBRT_GPU void operator*=(const RGB &c) {
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

PBRT_GPU inline RGB operator*(double scalar, const RGB &c) {
    return RGB(c.r * scalar, c.g * scalar, c.b * scalar);
}

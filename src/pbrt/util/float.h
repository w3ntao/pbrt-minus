#pragma once

#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/vector3.h"

static const double CPU_PI = std::acos(-1.0);

static constexpr double Infinity = std::numeric_limits<double>::infinity();

static constexpr double MachineEpsilon = std::numeric_limits<double>::epsilon() * 0.5;

constexpr double gamma(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

PBRT_CPU_GPU
inline double FMA(double a, double b, double c) {
    return std::fma(a, b, c);
}

PBRT_CPU_GPU Vector3f FMA(double a, const Vector3f &b, const Vector3f &c) {
    return {FMA(a, b.x, c.x), FMA(a, b.y, c.y), FMA(a, b.z, c.z)};
}

PBRT_CPU_GPU Vector3f FMA(const Vector3f &a, double b, const Vector3f &c) {
    return FMA(b, a, c);
}

template <typename Ta, typename Tb, typename Tc, typename Td>
PBRT_CPU_GPU auto difference_of_products(Ta a, Tb b, Tc c, Td d) {
    auto cd = c * d;
    auto diff = FMA(a, b, -cd);
    auto error = FMA(-c, d, cd);
    return diff + error;
}

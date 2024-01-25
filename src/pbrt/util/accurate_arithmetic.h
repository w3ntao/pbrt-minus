#pragma once

#include "pbrt/util/macro.h"

PBRT_CPU_GPU
static constexpr double compute_pi() {
    return acos(-1);
}

PBRT_CPU_GPU
static double degree_to_radian(double degree) {
    return compute_pi() / 180.0 * degree;
}

static constexpr double Infinity = std::numeric_limits<double>::infinity();

static constexpr double MachineEpsilon = std::numeric_limits<double>::epsilon() * 0.5;

PBRT_CPU_GPU
constexpr double gamma(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

PBRT_CPU_GPU
inline double FMA(double a, double b, double c) {
    return std::fma(a, b, c);
}

template <typename Ta, typename Tb, typename Tc, typename Td>
PBRT_CPU_GPU auto difference_of_products(Ta a, Tb b, Tc c, Td d) {
    auto cd = c * d;
    auto diff = FMA(a, b, -cd);
    auto error = FMA(-c, d, cd);
    return diff + error;
}

static constexpr double lerp(double x, double a, double b) {
    return (1 - x) * a + x * b;
}

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

template <typename T, std::enable_if_t<std::is_same_v<T, double>, bool> = true>
PBRT_CPU_GPU T FMA(T a, T b, T c) {
    return std::fma(a, b, c);
}

template <typename Ta, typename Tb, typename Tc, typename Td>
PBRT_CPU_GPU auto difference_of_products(Ta a, Tb b, Tc c, Td d) {
    auto cd = c * d;
    auto diff = FMA(a, b, -cd);
    auto error = FMA(-c, d, cd);
    return diff + error;
}

template <typename Ta, typename Tb, typename Tc, typename Td>
PBRT_CPU_GPU auto sum_of_products(Ta a, Tb b, Tc c, Td d) {
    auto cd = c * d;
    auto sum = FMA(a, b, cd);
    auto error = FMA(c, d, -cd);
    return sum + error;
}

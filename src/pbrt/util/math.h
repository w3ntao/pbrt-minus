#pragma once

#include "pbrt/util/macro.h"

static constexpr double Infinity = std::numeric_limits<double>::infinity();

static constexpr double MachineEpsilon = std::numeric_limits<double>::epsilon() * 0.5;

PBRT_CPU_GPU
constexpr double gamma(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

PBRT_CPU_GPU
static constexpr double compute_pi() {
    return acos(-1);
}

PBRT_CPU_GPU
static double degree_to_radian(double degree) {
    return compute_pi() / 180.0 * degree;
}

PBRT_CPU_GPU
constexpr double clamp(double x, double low, double high) {
    return x < low ? low : (x > high ? high : x);
}

PBRT_CPU_GPU
constexpr double clamp_0_1(double x) {
    return clamp(x, 0, 1);
}

PBRT_CPU_GPU inline double lerp(double x, double a, double b) {
    return (1 - x) * a + x * b;
}

PBRT_CPU_GPU constexpr double sqr(double v) {
    return v * v;
}

PBRT_CPU_GPU
inline double safe_sqrt(double x) {
    return std::sqrt(std::max(0., x));
}

PBRT_CPU_GPU constexpr double evaluate_polynomial(double t, double c) {
    return c;
}

template <typename... Args>
PBRT_CPU_GPU constexpr double evaluate_polynomial(double t, double c, Args... cRemaining) {
    return std::fma(t, evaluate_polynomial(t, cRemaining...), c);
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

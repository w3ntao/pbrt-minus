#pragma once

#include "pbrt/util/macro.h"

static constexpr double Infinity = std::numeric_limits<double>::infinity();

static constexpr double MachineEpsilon = std::numeric_limits<double>::epsilon() * 0.5;

PBRT_CPU_GPU
constexpr double gamma(int n) {
    return (n * MachineEpsilon) / (1 - n * MachineEpsilon);
}

PBRT_CPU_GPU
static double compute_pi() {
    return acos(-1);
}

PBRT_CPU_GPU
static double degree_to_radian(double degree) {
    return compute_pi() / 180.0 * degree;
}

template <typename T>
PBRT_CPU_GPU std::enable_if_t<std::is_floating_point_v<T>, bool> is_inf(T v) {
#if defined(__CUDA_ARCH__)
    return isinf(v);
#else
    return std::isinf(v);
#endif
}

PBRT_CPU_GPU
constexpr double clamp(double x, double low, double high) {
    return x < low ? low : (x > high ? high : x);
}

PBRT_CPU_GPU
constexpr double clamp_0_1(double x) {
    return clamp(x, 0, 1);
}

template <typename Predicate>
PBRT_CPU_GPU size_t find_interval(size_t sz, const Predicate &pred) {
    using ssize_t = std::make_signed_t<size_t>;
    ssize_t size = (ssize_t)sz - 2, first = 1;
    while (size > 0) {
        // Evaluate predicate at midpoint and update _first_ and _size_
        size_t half = (size_t)size >> 1, middle = first + half;
        bool predResult = pred(middle);
        first = predResult ? middle + 1 : first;
        size = predResult ? size - (half + 1) : half;
    }
    return (size_t)clamp((ssize_t)first - 1, 0, sz - 2);
}

PBRT_CPU_GPU inline double lerp(double x, double a, double b) {
    return (1 - x) * a + x * b;
}

PBRT_CPU_GPU constexpr double sqr(double v) {
    return v * v;
}

PBRT_CPU_GPU
inline double safe_sqrt(double x) {
    return std::sqrt(std::max(0.0, x));
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

template <typename T, std::enable_if_t<std::is_same_v<T, uint32_t>, bool> = true>
PBRT_CPU_GPU inline T left_shift3(T x) {
    if (x == (1 << 10)) {
        --x;
    }

    // clang-format off

    x = (x | (x << 16)) & 0b00000011000000000000000011111111;
    // x = ---- --98 ---- ---- ---- ---- 7654 3210
    x = (x | (x <<  8)) & 0b00000011000000001111000000001111;
    // x = ---- --98 ---- ---- 7654 ---- ---- 3210
    x = (x | (x <<  4)) & 0b00000011000011000011000011000011;
    // x = ---- --98 ---- 76-- --54 ---- 32-- --10
    x = (x | (x <<  2)) & 0b00001001001001001001001001001001;
    // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0

    // clang-format on

    return x;
}

template <typename T, std::enable_if_t<std::is_same_v<T, uint32_t>, bool> = true>
PBRT_CPU_GPU inline T encode_morton3(T x, T y, T z) {
    return (left_shift3(z) << 2) | (left_shift3(y) << 1) | left_shift3(x);
}

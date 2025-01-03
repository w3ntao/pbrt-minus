#pragma once

#include "pbrt/util/macro.h"
#include <limits>

#define _DoubleOneMinusEpsilon 0x1.fffffffffffffp-1
#define _FloatOneMinusEpsilon float(0x1.fffffep-1)

#ifdef PBRT_FLOAT_AS_DOUBLE
#define OneMinusEpsilon _DoubleOneMinusEpsilon
#else
#define OneMinusEpsilon _FloatOneMinusEpsilon
#endif

constexpr FloatType Infinity = std::numeric_limits<FloatType>::infinity();

constexpr FloatType MachineEpsilon = std::numeric_limits<FloatType>::epsilon() * 0.5;

// Mathematical Constants
constexpr FloatType ShadowEpsilon = 0.0001;

PBRT_CPU_GPU
constexpr FloatType gamma(int n) {
    return (FloatType(n) * MachineEpsilon) / (FloatType(1.0) - FloatType(n) * MachineEpsilon);
}

template <typename T>
PBRT_CPU_GPU inline T mod(T a, T b) {
    T result = a - (a / b) * b;
    return (T)((result < 0) ? result + b : result);
}

template <typename T>
PBRT_CPU_GPU constexpr T clamp(T x, T low, T high) {
    return x < low ? low : (x > high ? high : x);
}

PBRT_CPU_GPU
static FloatType compute_pi() {
    return acos(-1);
}

PBRT_CPU_GPU
static FloatType degree_to_radian(FloatType degree) {
    return compute_pi() / 180.0 * degree;
}

PBRT_CPU_GPU
inline FloatType safe_asin(FloatType x) {
    return std::asin(clamp<FloatType>(x, -1, 1));
}

PBRT_CPU_GPU
inline float safe_acos(float x) {
    return std::acos(clamp<FloatType>(x, -1, 1));
}

template <typename T>
PBRT_CPU_GPU std::enable_if_t<std::is_floating_point_v<T>, bool> is_inf(T v) {
#if defined(__CUDA_ARCH__)
    return isinf(v);
#else
    return std::isinf(v);
#endif
}

template <typename T>
PBRT_CPU_GPU T divide_and_ceil(T dividend, T divisor) {
    if (dividend % divisor == 0) {
        return dividend / divisor;
    }

    return dividend / divisor + 1;
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
    return clamp<size_t>((ssize_t)first - 1, 0, sz - 2);
}

PBRT_CPU_GPU
constexpr FloatType lerp(FloatType x, FloatType a, FloatType b) {
    return (1 - x) * a + x * b;
}

PBRT_CPU_GPU
constexpr FloatType sqr(FloatType v) {
    return v * v;
}

PBRT_CPU_GPU
static FloatType safe_sqrt(FloatType x) {
    return std::sqrt(std::max(FloatType(0.0), x));
}

PBRT_CPU_GPU constexpr FloatType evaluate_polynomial(FloatType t, FloatType c) {
    return c;
}

template <typename... Args>
PBRT_CPU_GPU constexpr FloatType evaluate_polynomial(FloatType t, FloatType c, Args... cRemaining) {
    return std::fma(t, evaluate_polynomial(t, cRemaining...), c);
}

template <typename T, std::enable_if_t<std::is_same_v<T, FloatType>, bool> = true>
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
PBRT_CPU_GPU constexpr T left_shift3(T x) {
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
PBRT_CPU_GPU constexpr T encode_morton3(T x, T y, T z) {
    return (left_shift3(z) << 2) | (left_shift3(y) << 1) | left_shift3(x);
}

PBRT_CPU_GPU
inline FloatType smooth_step(FloatType x, FloatType a, FloatType b) {
    if (a == b) {
        return (x < a) ? 0 : 1;
    }

    auto t = clamp<FloatType>((x - a) / (b - a), 0, 1);
    return t * t * (3 - 2 * t);
}

template <typename Func>
PBRT_CPU_GPU inline FloatType NewtonBisection(FloatType x0, FloatType x1, Func f,
                                              FloatType xEps = 1e-6f, FloatType fEps = 1e-6f) {
    // Check function endpoints for roots
    FloatType fx0 = f(x0).first;
    FloatType fx1 = f(x1).first;

    if (std::abs(fx0) < fEps) {
        return x0;
    }
    if (std::abs(fx1) < fEps) {
        return x1;
    }

    bool startIsNegative = fx0 < 0;

    // Set initial midpoint using linear approximation of _f_
    FloatType xMid = x0 + (x1 - x0) * -fx0 / (fx1 - fx0);

    while (true) {
        // Fall back to bisection if _xMid_ is out of bounds
        if (!(x0 < xMid && xMid < x1)) {
            xMid = (x0 + x1) / 2;
        }

        // Evaluate function and narrow bracket range _[x0, x1]_
        std::pair<FloatType, FloatType> fxMid = f(xMid);
        if (startIsNegative == (fxMid.first < 0)) {
            x0 = xMid;
        } else {
            x1 = xMid;
        }

        // Stop the iteration if converged
        if ((x1 - x0) < xEps || std::abs(fxMid.first) < fEps) {
            return xMid;
        }

        // Perform a Newton step
        xMid -= fxMid.first / fxMid.second;
    }
}

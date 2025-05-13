#pragma once

#include <cuda/std/tuple>
#include <limits>
#include <pbrt/gpu/macro.h>

#define _DoubleOneMinusEpsilon 0x1.fffffffffffffp-1
#define _FloatOneMinusEpsilon float(0x1.fffffep-1)

#ifdef PBRT_FLOAT_AS_DOUBLE
#define OneMinusEpsilon _DoubleOneMinusEpsilon
#else
#define OneMinusEpsilon _FloatOneMinusEpsilon
#endif

constexpr Real Sqrt2 = 1.41421356237309504880;

constexpr Real Infinity = std::numeric_limits<Real>::infinity();

constexpr Real MachineEpsilon = std::numeric_limits<Real>::epsilon() * 0.5;

// Mathematical Constants
constexpr Real ShadowEpsilon = 0.0001;

namespace pbrt {
PBRT_CPU_GPU
constexpr Real lerp(Real x, Real a, Real b) {
    return (1 - x) * a + x * b;
}
} // namespace pbrt

PBRT_CPU_GPU
constexpr Real gamma(int n) {
    return (Real(n) * MachineEpsilon) / (Real(1.0) - Real(n) * MachineEpsilon);
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
static Real compute_pi() {
    return acos(-1);
}

PBRT_CPU_GPU
static Real degree_to_radian(Real degree) {
    return compute_pi() / 180.0 * degree;
}

PBRT_CPU_GPU
inline Real safe_asin(Real x) {
    return std::asin(clamp<Real>(x, -1, 1));
}

PBRT_CPU_GPU
inline float safe_acos(float x) {
    return std::acos(clamp<Real>(x, -1, 1));
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
constexpr Real sqr(Real v) {
    return v * v;
}

PBRT_CPU_GPU
static Real safe_sqrt(Real x) {
    return std::sqrt(std::max(Real(0.0), x));
}

PBRT_CPU_GPU constexpr Real evaluate_polynomial(Real t, Real c) {
    return c;
}

template <typename... Args>
PBRT_CPU_GPU constexpr Real evaluate_polynomial(Real t, Real c, Args... cRemaining) {
    return std::fma(t, evaluate_polynomial(t, cRemaining...), c);
}

template <typename T, std::enable_if_t<std::is_same_v<T, Real>, bool> = true>
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
inline Real smooth_step(Real x, Real a, Real b) {
    if (a == b) {
        return (x < a) ? 0 : 1;
    }

    auto t = clamp<Real>((x - a) / (b - a), 0, 1);
    return t * t * (3 - 2 * t);
}

template <typename Func>
PBRT_CPU_GPU inline Real NewtonBisection(Real x0, Real x1, Func f,
                                              Real xEps = 1e-6f, Real fEps = 1e-6f) {
    // Check function endpoints for roots
    Real fx0 = f(x0).first;
    Real fx1 = f(x1).first;

    if (std::abs(fx0) < fEps) {
        return x0;
    }
    if (std::abs(fx1) < fEps) {
        return x1;
    }

    bool startIsNegative = fx0 < 0;

    // Set initial midpoint using linear approximation of _f_
    Real xMid = x0 + (x1 - x0) * -fx0 / (fx1 - fx0);

    while (true) {
        // Fall back to bisection if _xMid_ is out of bounds
        if (!(x0 < xMid && xMid < x1)) {
            xMid = (x0 + x1) / 2;
        }

        // Evaluate function and narrow bracket range _[x0, x1]_
        cuda::std::pair<Real, Real> fxMid = f(xMid);
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

PBRT_CPU_GPU
inline Real erf_inv(Real a) {
    /*
    erfinv:
         0 -> 0
         1 -> + inf
        -1 -> - inf
        if (x < -1 or x > 1) -> NAN

    for GPU code there is a non-zero probability it returns INF
    */

#if defined(__CUDA_ARCH__)
    return erfinv(a);
#else
    // https://stackoverflow.com/a/49743348
    Real p;
    Real t =
        std::log(std::max<Real>(std::fma(a, -a, 1), std::numeric_limits<Real>::min()));

    if (std::abs(t) > 6.125f) {              // maximum ulp error = 2.35793
        p = 3.03697567e-10f;                 //  0x1.4deb44p-32
        p = std::fma(p, t, 2.93243101e-8f);  //  0x1.f7c9aep-26
        p = std::fma(p, t, 1.22150334e-6f);  //  0x1.47e512p-20
        p = std::fma(p, t, 2.84108955e-5f);  //  0x1.dca7dep-16
        p = std::fma(p, t, 3.93552968e-4f);  //  0x1.9cab92p-12
        p = std::fma(p, t, 3.02698812e-3f);  //  0x1.8cc0dep-9
        p = std::fma(p, t, 4.83185798e-3f);  //  0x1.3ca920p-8
        p = std::fma(p, t, -2.64646143e-1f); // -0x1.0eff66p-2
        p = std::fma(p, t, 8.40016484e-1f);  //  0x1.ae16a4p-1
    } else {                                 // maximum ulp error = 2.35456
        p = 5.43877832e-9f;                  //  0x1.75c000p-28
        p = std::fma(p, t, 1.43286059e-7f);  //  0x1.33b458p-23
        p = std::fma(p, t, 1.22775396e-6f);  //  0x1.49929cp-20
        p = std::fma(p, t, 1.12962631e-7f);  //  0x1.e52bbap-24
        p = std::fma(p, t, -5.61531961e-5f); // -0x1.d70c12p-15
        p = std::fma(p, t, -1.47697705e-4f); // -0x1.35be9ap-13
        p = std::fma(p, t, 2.31468701e-3f);  //  0x1.2f6402p-9
        p = std::fma(p, t, 1.15392562e-2f);  //  0x1.7a1e4cp-7
        p = std::fma(p, t, -2.32015476e-1f); // -0x1.db2aeep-3
        p = std::fma(p, t, 8.86226892e-1f);  //  0x1.c5bf88p-1
    }

    return a * p;
#endif
}

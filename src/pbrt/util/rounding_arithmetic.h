#pragma once

#include <cstring>

template <class To, class From>
PBRT_CPU_GPU
    typename std::enable_if_t<sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> &&
                                  std::is_trivially_copyable_v<To>,
                              To>
    bit_cast(const From &src) noexcept {
    static_assert(std::is_trivially_constructible_v<To>,
                  "This implementation requires the destination type to be trivially "
                  "constructible");
    To dst;
    std::memcpy(&dst, &src, sizeof(To));
    return dst;
}

PBRT_CPU_GPU
inline uint64_t FloatToBits(double f) {
#if defined(__CUDA_ARCH__)
    return __double_as_longlong(f);
#else
    return bit_cast<uint64_t>(f);
#endif
}

PBRT_CPU_GPU
inline double BitsToFloat(uint64_t ui) {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(ui);
#else
    return bit_cast<double>(ui);
#endif
}

PBRT_CPU_GPU
inline double NextFloatUp(double v) {
    if (std::isinf(v) && v > 0.) {
        return v;
    }

    if (v == -0.f) {
        v = 0.f;
    }

    uint64_t ui = FloatToBits(v);
    if (v >= 0.) {
        ++ui;
    } else {
        --ui;
    }

    return BitsToFloat(ui);
}

PBRT_CPU_GPU
inline float NextFloatDown(float v) {
    // Handle infinity and positive zero for _NextFloatDown()_
    if (std::isinf(v) && v < 0.) {
        return v;
    }

    if (v == 0.f) {
        v = -0.f;
    }

    uint32_t ui = FloatToBits(v);
    if (v > 0) {
        --ui;
    } else {
        ++ui;
    }

    return BitsToFloat(ui);
}

PBRT_CPU_GPU double add_round_up(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_ru(a, b);
#else
    return NextFloatUp(a + b);
#endif
}

PBRT_CPU_GPU double add_round_down(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rd(a, b);
#else
    return NextFloatDown(a + b);
#endif
}

PBRT_CPU_GPU double sub_round_up(double a, double b) {
    return add_round_up(a, -b);
}
PBRT_CPU_GPU double sub_round_down(double a, double b) {
    return add_round_down(a, -b);
}

PBRT_CPU_GPU double mul_round_up(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_ru(a, b);
#else
    return NextFloatUp(a * b);
#endif
}

PBRT_CPU_GPU double mul_round_down(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rd(a, b);
#else
    return NextFloatDown(a * b);
#endif
}

PBRT_CPU_GPU double div_round_up(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __ddiv_ru(a, b);
#else
    return NextFloatUp(a / b);
#endif
}
PBRT_CPU_GPU double div_round_down(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __ddiv_rd(a, b);
#else
    return NextFloatDown(a / b);
#endif
}

PBRT_CPU_GPU double sqrt_round_up(double a) {
#if defined(__CUDA_ARCH__)
    return __dsqrt_ru(a);
#else
    return NextFloatUp(std::sqrt(a));
#endif
}

PBRT_CPU_GPU double sqrt_round_down(double a) {
#if defined(__CUDA_ARCH__)
    return __dsqrt_rd(a);
#else
    return NextFloatDown(std::sqrt(a));
#endif
}

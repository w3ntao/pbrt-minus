#pragma once

#include <cstring>
#include "pbrt/util/macro.h"

template <class To, class From>
PBRT_CPU_GPU std::enable_if_t<sizeof(To) == sizeof(From) && std::is_trivially_copyable_v<From> &&
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

template <typename T, std::enable_if_t<std::is_same_v<T, double>, bool> = true>
PBRT_CPU_GPU uint64_t float_to_bits(T f) {
#if defined(__CUDA_ARCH__)
    return __double_as_longlong(f);
#else
    return bit_cast<uint64_t>(f);
#endif
}

template <typename T, std::enable_if_t<std::is_same_v<T, uint64_t>, bool> = true>
PBRT_CPU_GPU double bits_to_float(T ui) {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(ui);
#else
    return bit_cast<double>(ui);
#endif
}

template <typename T, std::enable_if_t<std::is_same_v<T, double>, bool> = true>
PBRT_CPU_GPU T next_float_up(T v) {
    if (std::isinf(v) && v > 0.0) {
        return v;
    }

    if (v == -0.0) {
        v = 0.0;
    }

    uint64_t ui = float_to_bits(v);
    if (v >= 0.0) {
        ++ui;
    } else {
        --ui;
    }

    return bits_to_float(ui);
}

template <typename T, std::enable_if_t<std::is_same_v<T, double>, bool> = true>
PBRT_CPU_GPU T next_float_down(T v) {
    // Handle infinity and positive zero
    if (std::isinf(v) && v < 0.0) {
        return v;
    }

    if (v == 0.0) {
        v = -0.0;
    }

    uint64_t ui = float_to_bits(v);
    if (v > 0) {
        --ui;
    } else {
        ++ui;
    }

    return bits_to_float(ui);
}

template <typename T, std::enable_if_t<std::is_same_v<T, double>, bool> = true>
PBRT_CPU_GPU T add_round_up(T a, T b) {
#if defined(__CUDA_ARCH__)
    return __dadd_ru(a, b);
#else
    return next_float_up(a + b);
#endif
}

template <typename T, std::enable_if_t<std::is_same_v<T, double>, bool> = true>
PBRT_CPU_GPU T add_round_down(T a, T b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rd(a, b);
#else
    return next_float_down(a + b);
#endif
}

template <typename T, std::enable_if_t<std::is_same_v<T, double>, bool> = true>
PBRT_CPU_GPU T sub_round_up(T a, T b) {
    return add_round_up(a, -b);
}

template <typename T, std::enable_if_t<std::is_same_v<T, double>, bool> = true>
PBRT_CPU_GPU T sub_round_down(T a, T b) {
    return add_round_down(a, -b);
}

template <typename T, std::enable_if_t<std::is_same_v<T, double>, bool> = true>
PBRT_CPU_GPU T mul_round_up(T a, T b) {
#if defined(__CUDA_ARCH__)
    return __dmul_ru(a, b);
#else
    return next_float_up(a * b);
#endif
}

template <typename T, std::enable_if_t<std::is_same_v<T, double>, bool> = true>
PBRT_CPU_GPU T mul_round_down(T a, T b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rd(a, b);
#else
    return next_float_down(a * b);
#endif
}

template <typename T, std::enable_if_t<std::is_same_v<T, double>, bool> = true>
PBRT_CPU_GPU T div_round_up(T a, T b) {
#if defined(__CUDA_ARCH__)
    return __ddiv_ru(a, b);
#else
    return next_float_up(a / b);
#endif
}

template <typename T, std::enable_if_t<std::is_same_v<T, double>, bool> = true>
PBRT_CPU_GPU T div_round_down(T a, T b) {
#if defined(__CUDA_ARCH__)
    return __ddiv_rd(a, b);
#else
    return next_float_down(a / b);
#endif
}

template <typename T, std::enable_if_t<std::is_same_v<T, double>, bool> = true>
PBRT_CPU_GPU T sqrt_round_up(T a) {
#if defined(__CUDA_ARCH__)
    return __dsqrt_ru(a);
#else
    return next_float_up(std::sqrt(a));
#endif
}

template <typename T, std::enable_if_t<std::is_same_v<T, double>, bool> = true>
PBRT_CPU_GPU T sqrt_round_down(T a) {
#if defined(__CUDA_ARCH__)
    return __dsqrt_rd(a);
#else
    return next_float_down(std::sqrt(a));
#endif
}

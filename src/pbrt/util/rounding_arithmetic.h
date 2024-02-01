#pragma once

#include <cstring>

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
inline double next_float_up(double v) {
    if (std::isinf(v) && v > 0.0) {
        return v;
    }

    if (v == -0.0) {
        v = 0.0;
    }

    uint64_t ui = FloatToBits(v);
    if (v >= 0.0) {
        ++ui;
    } else {
        --ui;
    }

    return BitsToFloat(ui);
}

template <class T>
PBRT_CPU_GPU
double next_float_up(T) = delete; // prevent non double argument passed in

PBRT_CPU_GPU
inline double next_float_down(double v) {
    // Handle infinity and positive zero
    if (std::isinf(v) && v < 0.0) {
        return v;
    }

    if (v == 0.0) {
        v = -0.0;
    }

    uint64_t ui = FloatToBits(v);
    if (v > 0) {
        --ui;
    } else {
        ++ui;
    }

    return BitsToFloat(ui);
}

template <class T>
PBRT_CPU_GPU
double next_float_down(T) = delete; // prevent non double argument passed in

PBRT_CPU_GPU inline double add_round_up(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_ru(a, b);
#else
    return next_float_up(a + b);
#endif
}

PBRT_CPU_GPU inline double add_round_down(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rd(a, b);
#else
    return next_float_down(a + b);
#endif
}

PBRT_CPU_GPU inline double sub_round_up(double a, double b) {
    return add_round_up(a, -b);
}
PBRT_CPU_GPU inline double sub_round_down(double a, double b) {
    return add_round_down(a, -b);
}

PBRT_CPU_GPU inline double mul_round_up(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_ru(a, b);
#else
    return next_float_up(a * b);
#endif
}

PBRT_CPU_GPU inline double mul_round_down(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rd(a, b);
#else
    return next_float_down(a * b);
#endif
}

PBRT_CPU_GPU inline double div_round_up(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __ddiv_ru(a, b);
#else
    return next_float_up(a / b);
#endif
}
PBRT_CPU_GPU inline double div_round_down(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __ddiv_rd(a, b);
#else
    return next_float_down(a / b);
#endif
}

PBRT_CPU_GPU inline double sqrt_round_up(double a) {
#if defined(__CUDA_ARCH__)
    return __dsqrt_ru(a);
#else
    return next_float_up(std::sqrt(a));
#endif
}

PBRT_CPU_GPU inline double sqrt_round_down(double a) {
#if defined(__CUDA_ARCH__)
    return __dsqrt_rd(a);
#else
    return next_float_down(std::sqrt(a));
#endif
}

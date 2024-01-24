#pragma once

#include "pbrt/util/macro.h"

struct CompensatedFloat {
    // CompensatedFloat Public Methods
    PBRT_CPU_GPU
    explicit CompensatedFloat(double _v, double _err = 0) : v(_v), err(_err) {}

    PBRT_CPU_GPU
    explicit operator double() const {
        return v + err;
    }

    double v, err;
};

PBRT_CPU_GPU inline CompensatedFloat two_prod(double a, double b) {
    double ab = a * b;
    return CompensatedFloat(ab, std::fma(a, b, -ab));
}

PBRT_CPU_GPU inline CompensatedFloat two_sum(double a, double b) {
    double s = a + b, delta = s - a;
    return CompensatedFloat(s, (a - (s - delta)) + (b - delta));
}

namespace internal {
// InnerProduct Helper Functions
template <typename Float>
PBRT_CPU_GPU CompensatedFloat inner_product(Float a, Float b) {
    return two_prod(a, b);
}

// Accurate dot products with FMA: Graillat et al.,
// https://www-pequan.lip6.fr/~graillat/papers/posterRNC7.pdf
//
// Accurate summation, dot product and polynomial evaluation in complex
// floating point arithmetic, Graillat and Menissier-Morain.
template <typename Float, typename... T>
PBRT_CPU_GPU CompensatedFloat inner_product(Float a, Float b, T... terms) {
    CompensatedFloat ab = two_prod(a, b);
    CompensatedFloat tp = inner_product(terms...);
    CompensatedFloat sum = two_sum(ab.v, tp.v);
    return CompensatedFloat(sum.v, ab.err + (tp.err + sum.err));
}

} // namespace internal

template <typename... T>
PBRT_CPU_GPU std::enable_if_t<std::conjunction_v<std::is_arithmetic<T>...>, double>
inner_product(T... terms) {
    CompensatedFloat ip = internal::inner_product(terms...);
    return double(ip);
}

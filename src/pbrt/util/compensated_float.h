#pragma once

#include "pbrt/util/macro.h"

struct CompensatedFloat {
  public:
    // CompensatedFloat Public Methods
    PBRT_CPU_GPU
    CompensatedFloat(double _v, double _err = 0) : v(_v), err(_err) {}

    PBRT_CPU_GPU
    explicit operator double() const {
        return v + err;
    }

    double v, err;
};

PBRT_CPU_GPU inline CompensatedFloat TwoProd(double a, double b) {
    double ab = a * b;
    return {ab, std::fma(a, b, -ab)};
}

PBRT_CPU_GPU inline CompensatedFloat TwoSum(double a, double b) {
    double s = a + b, delta = s - a;
    return {s, (a - (s - delta)) + (b - delta)};
}

namespace internal {
// InnerProduct Helper Functions
template <typename Float>
PBRT_CPU_GPU inline CompensatedFloat InnerProduct(Float a, Float b) {
    return TwoProd(a, b);
}

// Accurate dot products with FMA: Graillat et al.,
// https://www-pequan.lip6.fr/~graillat/papers/posterRNC7.pdf
//
// Accurate summation, dot product and polynomial evaluation in complex
// floating point arithmetic, Graillat and Menissier-Morain.
template <typename Float, typename... T>
PBRT_CPU_GPU CompensatedFloat InnerProduct(Float a, Float b, T... terms) {
    CompensatedFloat ab = TwoProd(a, b);
    CompensatedFloat tp = InnerProduct(terms...);
    CompensatedFloat sum = TwoSum(ab.v, tp.v);
    return {sum.v, ab.err + (tp.err + sum.err)};
}

} // namespace internal

template <typename... T>
PBRT_CPU_GPU std::enable_if_t<std::conjunction_v<std::is_arithmetic<T>...>, double>
InnerProduct(T... terms) {
    CompensatedFloat ip = internal::InnerProduct(terms...);
    return double(ip);
}

#pragma once

#include <pbrt/gpu/macro.h>

namespace pbrt {

template <typename T>
PBRT_CPU_GPU inline void swap(T &a, T &b) {
    T tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}

PBRT_CPU_GPU inline float copysign(float mag, float sign) {
#ifdef PBRT_IS_GPU_CODE
    return ::copysignf(mag, sign);
#else
    return std::copysign(mag, sign);
#endif
}

PBRT_CPU_GPU inline double copysign(double mag, double sign) {
#ifdef PBRT_IS_GPU_CODE
    return ::copysign(mag, sign);
#else
    return std::copysign(mag, sign);
#endif
}

template <int n>
PBRT_CPU_GPU constexpr float pow(float v) {
    if constexpr (n < 0) {
        return 1 / pow<-n>(v);
    }

    float n2 = pow<n / 2>(v);
    return n2 * n2 * pow<n & 1>(v);
}

template <>
PBRT_CPU_GPU constexpr float pow<1>(float v) {
    return v;
}
template <>
PBRT_CPU_GPU constexpr float pow<0>(float v) {
    return 1;
}

} // namespace pbrt

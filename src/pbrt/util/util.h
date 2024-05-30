#pragma once

#include "pbrt/util/macro.h"

namespace pstd {

template <typename T>
PBRT_CPU_GPU inline void swap(T &a, T &b) {
    T tmp = std::move(a);
    a = std::move(b);
    b = std::move(tmp);
}

PBRT_CPU_GPU
inline double copysign(double mag, double sign) {
#if defined(__CUDA_ARCH__)
    return ::copysign(mag, sign);
#else
    return std::copysign(mag, sign);
#endif
}
} // namespace pstd
// namespace pstd
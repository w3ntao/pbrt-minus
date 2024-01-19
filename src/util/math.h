#pragma once

#include "util/macro.h"

inline double clamp_0_1(double x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

PBRT_CPU_GPU inline double gpu_clamp_0_1(double x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

PBRT_CPU_GPU constexpr double sqr(double v) {
    return v * v;
}

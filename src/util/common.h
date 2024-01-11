#pragma once

#define PBRT_CPU_GPU __host__ __device__
#define PBRT_GPU __device__

const double CPU_PI = std::acos(-1.0);

inline double clamp_0_1(double x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

PBRT_CPU_GPU inline double gpu_clamp_0_1(double x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

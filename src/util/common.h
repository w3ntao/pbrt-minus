#pragma once

const double CPU_PI = std::acos(-1.0);

inline double clamp_0_1(double x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

__host__ __device__ inline double gpu_clamp_0_1(double x) {
    return x < 0.0 ? 0.0 : (x > 1.0 ? 1.0 : x);
}

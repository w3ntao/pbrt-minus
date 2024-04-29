#pragma once

#include <curand_kernel.h>

#include "pbrt/euclidean_space/point2.h"

class IndependentSampler {
  public:
    PBRT_CPU_GPU
    void init(uint _samples_per_pixel) {
        samples_per_pixel = _samples_per_pixel;
    }

    PBRT_GPU
    void start_pixel_sample(const uint pixel_idx, const uint sample_idx, const uint dimension) {
        curand_init(pixel_idx, sample_idx, dimension, &rand_state);
    }

    PBRT_CPU_GPU
    uint get_samples_per_pixel() const {
        return samples_per_pixel;
    }

    PBRT_GPU FloatType get_1d() {
        return curand_uniform(&rand_state);
    }

    PBRT_GPU Point2f get_2d() {
        return Point2f(curand_uniform(&rand_state), curand_uniform(&rand_state));
    }

    PBRT_GPU Point2f get_pixel_2d() {
        return get_2d();
    }

  private:
    curandState rand_state;
    uint samples_per_pixel;
};

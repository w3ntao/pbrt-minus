#pragma once

#include <curand_kernel.h>

#include "pbrt/base/sampler.h"
#include "pbrt/euclidean_space/point2.h"

class IndependentSampler : public Sampler {
  private:
    curandState rand_state;

  public:
    PBRT_GPU IndependentSampler(int seed) {
        curand_init(seed, 0, 0, &rand_state);
    }

    ~IndependentSampler() override = default;

    PBRT_GPU double get_1d() override {
        return curand_uniform(&rand_state);
    }

    PBRT_GPU Point2f get_2d() override {
        return Point2f(curand_uniform(&rand_state), curand_uniform(&rand_state));
    }
};

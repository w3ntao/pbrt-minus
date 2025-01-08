#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/util/hash.h"
#include "pbrt/util/rng.h"

class IndependentSampler {
  public:
    PBRT_CPU_GPU
    void init(uint _samples_per_pixel) {
        samples_per_pixel = _samples_per_pixel;
    }

    PBRT_CPU_GPU
    void start_pixel_sample(const uint pixel_idx, const uint sample_idx, const uint dimension) {
        rng.set_sequence(pbrt::hash(pixel_idx));
        rng.advance(sample_idx * 65536ull + dimension);
    }

    PBRT_CPU_GPU
    uint get_samples_per_pixel() const {
        return samples_per_pixel;
    }

    PBRT_CPU_GPU FloatType get_1d() {
        return rng.uniform<FloatType>();
    }

    PBRT_CPU_GPU Point2f get_2d() {
        return Point2f(rng.uniform<FloatType>(), rng.uniform<FloatType>());
    }

    PBRT_CPU_GPU Point2f get_pixel_2d() {
        return get_2d();
    }

  private:
    RNG rng;
    uint samples_per_pixel;
};

#pragma once

#include <pbrt/euclidean_space/point2.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/rng.h>

class IndependentSampler {
  public:
    PBRT_CPU_GPU
    IndependentSampler(int _samples_per_pixel) : samples_per_pixel(_samples_per_pixel) {}

    PBRT_CPU_GPU
    void start_pixel_sample(const int pixel_idx, const int sample_idx, const int dimension) {
        rng.set_sequence(pbrt::hash(pixel_idx));
        rng.advance(sample_idx * 65536ull + dimension);
    }

    PBRT_CPU_GPU
    int get_samples_per_pixel() const {
        return samples_per_pixel;
    }

    PBRT_CPU_GPU Real get_1d() {
        return rng.uniform<Real>();
    }

    PBRT_CPU_GPU Point2f get_2d() {
        return Point2f(rng.uniform<Real>(), rng.uniform<Real>());
    }

    PBRT_CPU_GPU Point2f get_pixel_2d() {
        return get_2d();
    }

  private:
    RNG rng;
    int samples_per_pixel = 0;
};

#pragma once

#include <vector>
#include <curand_kernel.h>

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/util/hash.h"
#include "pbrt/util/permutation.h"

class StratifiedSampler {
  public:
    PBRT_CPU_GPU
    void init(uint _samples_per_dimension) {
        samples_per_dimension = _samples_per_dimension;
    }

    PBRT_CPU_GPU
    uint get_samples_per_pixel() const {
        return samples_per_dimension * samples_per_dimension;
    }

    PBRT_GPU
    void start_pixel_sample(const uint _pixel_idx, const uint _sample_idx, const uint _dimension) {
        pixel_idx = _pixel_idx;
        sample_idx = _sample_idx;
        dimension = _dimension;

        curand_init(0, pstd::hash(pixel_idx), sample_idx * 65536ull + dimension, &rand_state);
    }

    PBRT_GPU FloatType get_1d() {
        uint64_t hash = pstd::hash(pixel_idx, dimension);
        int stratum = permutation_element(sample_idx, get_samples_per_pixel(), hash);

        dimension += 1;
        auto delta = curand_uniform(&rand_state);
        return (stratum + delta) / get_samples_per_pixel();
    }

    PBRT_GPU Point2f get_2d() {
        uint64_t hash = pstd::hash(pixel_idx, dimension);
        int stratum = permutation_element(sample_idx, get_samples_per_pixel(), hash);
        dimension += 2;

        auto x = stratum % samples_per_dimension;
        auto y = stratum / samples_per_dimension;

        auto delta_x = curand_uniform(&rand_state);
        auto delta_y = curand_uniform(&rand_state);

        return Point2f((x + delta_x) / samples_per_dimension,
                       (y + delta_y) / samples_per_dimension);
    }

    PBRT_GPU Point2f get_pixel_2d() {
        return get_2d();
    }

  private:
    // StratifiedSampler Private Members
    uint samples_per_dimension;

    curandState rand_state;
    uint pixel_idx;
    uint sample_idx;
    uint dimension;
};

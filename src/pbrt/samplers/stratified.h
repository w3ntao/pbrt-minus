#pragma once

#include <pbrt/euclidean_space/point2.h>
#include <pbrt/util/hash.h>
#include <pbrt/util/permutation.h>
#include <pbrt/util/rng.h>

class StratifiedSampler {
  public:
    PBRT_CPU_GPU
    StratifiedSampler(uint samples_per_pixel)
        : samples_per_dimension(static_cast<uint>(std::sqrt(samples_per_pixel))) {
        if (sqr(samples_per_dimension) != samples_per_pixel) {
            REPORT_FATAL_ERROR();
        }
    }

    PBRT_CPU_GPU
    uint get_samples_per_pixel() const {
        return samples_per_dimension * samples_per_dimension;
    }

    PBRT_CPU_GPU
    void start_pixel_sample(const uint _pixel_idx, const uint _sample_idx, const uint _dimension) {
        pixel_idx = _pixel_idx;
        sample_idx = _sample_idx;
        dimension = _dimension;

        rng.set_sequence(pbrt::hash(_pixel_idx));
        rng.advance(_sample_idx * 65536ull + _dimension);
    }

    PBRT_CPU_GPU Real get_1d() {
        uint64_t hash = pbrt::hash(pixel_idx, dimension);
        int stratum = permutation_element(sample_idx, get_samples_per_pixel(), hash);

        dimension += 1;

        auto delta = rng.uniform<Real>();

        return (stratum + delta) / get_samples_per_pixel();
    }

    PBRT_CPU_GPU Point2f get_2d() {
        uint64_t hash = pbrt::hash(pixel_idx, dimension);
        int stratum = permutation_element(sample_idx, get_samples_per_pixel(), hash);
        dimension += 2;

        auto x = stratum % samples_per_dimension;
        auto y = stratum / samples_per_dimension;

        auto delta_x = rng.uniform<Real>();
        auto delta_y = rng.uniform<Real>();

        return Point2f((x + delta_x) / samples_per_dimension,
                       (y + delta_y) / samples_per_dimension);
    }

    PBRT_CPU_GPU Point2f get_pixel_2d() {
        return get_2d();
    }

  private:
    uint samples_per_dimension;

    RNG rng;
    uint pixel_idx = 0;
    uint sample_idx = 0;
    uint dimension = 0;
};

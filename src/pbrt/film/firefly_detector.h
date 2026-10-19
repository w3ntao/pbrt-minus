#pragma once

#include "pbrt/gpu/macro.h"

#include <bit>
#include <cmath>

namespace firefly {

constexpr Real alpha = Real(0.5);
constexpr int beta = 2;
constexpr Real max_luminance = Real(1000);
static_assert(beta > 1 && (beta & (beta - 1)) == 0, "firefly::beta must be a power of two greater than one");
constexpr int beta_log2 = std::countr_zero(static_cast<unsigned int>(beta));
static_assert(alpha > 0 && max_luminance > alpha);

// Use the actual number of buffers for m in tau, including both endpoints.
// With alpha=1/2 and beta=2, twelve buffers cover centers 0.5 through 1024.
constexpr int layer_count = [] {
    int count = 1;
    Real center = alpha;
    while (center < max_luminance) {
        center *= beta;
        ++count;
    }
    return count;
}();

PBRT_CPU_GPU
inline Real tau(const int samples_per_pixel) {
    const Real uniform_count = Real(samples_per_pixel) / Real(layer_count);

    return std::max<Real>(3, uniform_count);
}

struct Sample {
    int lower_layer = -1;
    Real lower_count = 0;
    Real upper_count = 0;
    bool forced_outlier = false;
};

PBRT_CPU_GPU
inline Sample split(Real luminance) {
    if (!(luminance > 0) || !std::isfinite(luminance)) {
        return {};
    }

    // Section 6.1 does not specify out-of-range splatting. Keep one count
    // per positive finite sample by clamping low/high values to endpoint buffers.
    if (luminance < alpha) {
        return {0, 1, 0, false};
    }
    if (luminance > max_luminance) {
        return {layer_count - 2, 0, 1, true};
    }

    const Real coordinate = std::log2(luminance / alpha) / Real(beta_log2);
    const int lower_layer = static_cast<int>(coordinate);
    // Follow the printed weights in Bitterli and Jarosz 2019, Section 6.1:
    // w = frac(log_beta(x/alpha)) goes to B_j, and 1-w to B_(j+1).
    // This deliberately differs from the usual continuous interpolation direction.
    const Real weight = coordinate - lower_layer;
    return {lower_layer, weight, Real(1) - weight, false};
}

struct WeightRange {
    Real min_weight = 1;
    Real max_weight = -1;

    PBRT_CPU_GPU
    void record(const Real weight) {
        min_weight = weight < min_weight ? weight : min_weight;
        max_weight = weight > max_weight ? weight : max_weight;
    }

    PBRT_CPU_GPU
    bool empty() const {
        return min_weight > max_weight;
    }
};

PBRT_CPU_GPU
inline Real interpolated_count(const Real lower_count, const Real upper_count, const Real lower_weight) {
    // A single affine expression keeps the query monotone in the observed weight.
    return std::fma(lower_weight, lower_count - upper_count, upper_count);
}

PBRT_CPU_GPU
inline Real sample_count(Real *const *counts, const int pixel_index, const Sample &sample) {
    if (sample.lower_layer < 0) {
        return 0;
    }
    return interpolated_count(counts[sample.lower_layer][pixel_index], counts[sample.lower_layer + 1][pixel_index],
                              sample.lower_count);
}

PBRT_CPU_GPU
inline bool is_outlier(Real *const *counts, const int pixel_index, const Sample &sample, const Real threshold) {
    return sample.forced_outlier || (sample.lower_layer >= 0 && sample_count(counts, pixel_index, sample) < threshold);
}

PBRT_CPU_GPU
inline bool is_candidate(Real *const *counts, WeightRange *const *ranges, const int pixel_index, const Real threshold,
                         const bool forced_outlier = false) {
    if (forced_outlier) {
        return true;
    }
    for (int layer = 0; layer < layer_count - 1; ++layer) {
        const auto &range = ranges[layer][pixel_index];
        if (range.empty()) {
            continue;
        }
        const Real lower_count = counts[layer][pixel_index];
        const Real upper_count = counts[layer + 1][pixel_index];
        // On a frozen histogram the query is affine within each log interval.
        // Its minimum over all actual samples is attained at an observed endpoint.
        if (interpolated_count(lower_count, upper_count, range.min_weight) < threshold ||
            interpolated_count(lower_count, upper_count, range.max_weight) < threshold) {
            return true;
        }
    }

    return false;
}

} // namespace firefly

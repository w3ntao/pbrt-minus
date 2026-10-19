#pragma once

#include "pbrt/gpu/macro.h"
#include "pbrt/spectrum_util/spectrum_constants_cie.h"

#include <bit>
#include <cmath>

namespace firefly {
/*
    originally from https://cg.ivd.kit.edu/publications/2018/rwmc/reweighting-fireflies-preprint.pdf
    parameters configured with https://benedikt-bitterli.me/smlt/bitterli19selectively.pdf
*/

// Match SampledSpectrum::y() units for the first log-luminance center.
constexpr Real alpha = Real(0.5) / CIE_Y_integral;
constexpr int beta = 2;
constexpr int layer_count = 10;
static_assert(beta > 1 && (beta & (beta - 1)) == 0, "firefly::beta must be a power of two greater than one");
static_assert(layer_count >= 2, "firefly cascade requires at least two layers");
constexpr int beta_log2 = std::countr_zero(static_cast<unsigned int>(beta));

// The final layer center sets the cutoff: alpha * beta^(layer_count - 1).
constexpr Real max_luminance = [] {
    Real center = alpha;
    for (int layer = 1; layer < layer_count; ++layer) {
        center *= beta;
    }
    return center;
}();
static_assert(alpha > 0 && max_luminance > alpha);

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
    // The final center (or a value rounded to it) has no following layer.
    if (coordinate >= layer_count - 1) {
        return {layer_count - 2, 0, 1, false};
    }
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

#pragma once

#include <cstdint>
#include <limits>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/spectrum_util/rgb.h>

class GPUMemoryAllocator;

namespace statmc {

struct PixelStatistics {
    uint32_t n = 0;
    // Joint moments of A = weight * XYZ and W = weight. Unit weights retain
    // the ordinary XYZ sample-mean / squared-deviation semantics.
    RGB mean{0};
    RGB m2{0};
    Real mean_weight = 1;
    Real m2_weight = 0;
    RGB covariance{0};

    // One complete camera sample, including zero contributions, is one update.
    // A pixel must have a serialized owner; this operation uses no atomics.
    PBRT_CPU_GPU
    void add(const RGB &xyz, const Real weight = 1) {
        if (n == UINT32_MAX) {
            invalidate();
            return;
        }
        ++n;
        const RGB contribution = weight == 0 ? RGB(0) : xyz * weight;
        if (!std::isfinite(weight) || contribution.has_nan() || mean.has_nan() || m2.has_nan() ||
            !std::isfinite(mean_weight) || !std::isfinite(m2_weight) || covariance.has_nan()) {
            // Do not silently omit bad samples and lend biased statistics to
            // neighboring pixels. Invalid statistics remain invalid forever.
            invalidate();
            return;
        }
        if (n == 1) {
            mean = contribution;
            mean_weight = weight;
            return;
        }
        const RGB delta = contribution - mean;
        const Real delta_weight = weight - mean_weight;
        mean += delta / Real(n);
        mean_weight += delta_weight / Real(n);
        m2 += delta * (contribution - mean);
        m2_weight += delta_weight * (weight - mean_weight);
        covariance += delta * (weight - mean_weight);
        if (mean.has_nan() || m2.has_nan() || !std::isfinite(mean_weight) || !std::isfinite(m2_weight) ||
            covariance.has_nan()) {
            invalidate();
            return;
        }
        // Welford's accumulated squared deviations are nonnegative; remove
        // possible negative roundoff without changing a positive variance.
        for (uint8_t c = 0; c < 3; ++c) {
            if (m2[c] < 0) {
                m2[c] = 0;
            }
        }
        if (m2_weight < 0) {
            m2_weight = 0;
        }
    }

    // Film estimates sum(w * XYZ) / sum(w), rather than the mean of XYZ.
    // The delta method estimates its variance from the centered residual
    // A - ratio * W. This remains an approximation for signed weights and
    // stratified samples, rather than an exact Student-t sampling model.
    PBRT_CPU_GPU
    bool estimate(RGB &xyz_mean, RGB &mean_variance) const {
        if (n < 2 || mean.has_nan() || m2.has_nan() || covariance.has_nan() ||
            !std::isfinite(mean_weight) || !std::isfinite(m2_weight) || m2_weight < 0 || mean_weight == 0) {
            return false;
        }
        const Real epsilon = std::numeric_limits<Real>::epsilon();
        const Real sigma_weight = std::sqrt(m2_weight / Real(n - 1));
        // Reject cancellation relative to the weights' scale; uniformly
        // rescaled or entirely negative weights remain supported. A later
        // sample may restore a usable denominator, so this is not permanent.
        if (std::abs(mean_weight) <= std::sqrt(epsilon) * sigma_weight) {
            return false;
        }
        xyz_mean = mean / mean_weight;
        if (xyz_mean.has_nan()) {
            return false;
        }
        for (uint8_t c = 0; c < 3; ++c) {
            if (m2[c] < 0) {
                return false;
            }
            const Real ratio = xyz_mean[c];
            const Real cross = ratio * covariance[c];
            const Real weight_term = ratio * (ratio * m2_weight);
            Real residual = std::fma(ratio, ratio * m2_weight, std::fma(-ratio, covariance[c], m2[c])) - cross;
            if (!std::isfinite(residual) || !std::isfinite(cross) || !std::isfinite(weight_term)) {
                return false;
            }
            const Real scale = std::max(m2[c], std::max(std::abs(cross), std::abs(weight_term)));
            if (residual < 0) {
                if (-residual > 32 * epsilon * scale) {
                    return false;
                }
                residual = 0;
            }
            mean_variance[c] = residual / Real(n - 1) / Real(n) / mean_weight / mean_weight;
        }
        return !mean_variance.has_nan();
    }

  private:
    PBRT_CPU_GPU
    void invalidate() {
        mean = m2 = covariance = RGB(NAN);
        mean_weight = m2_weight = NAN;
    }
};

// The allocator owns the arrays and must outlive this object and its users.
class Denoiser {
  public:
    Denoiser(Point2i resolution, GPUMemoryAllocator &allocator);

    PixelStatistics *statistics() const {
        return pixel_statistics;
    }

    // The input must be GPU-accessible, and rendering/statistics updates must
    // be finished. Returns a synchronized, separately allocated managed image.
    const RGB *denoise(const RGB *linear_rgb);

  private:
    Point2i resolution;
    PixelStatistics *pixel_statistics = nullptr;
    RGB *output = nullptr;
    Real *critical_values = nullptr;
};

} // namespace statmc

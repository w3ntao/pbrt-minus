#pragma once

#include <cstdint>
#include <limits>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/spectrum_util/rgb.h>

class GPUMemoryAllocator;
struct GBufferPixel;

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

struct TransformedStatistics {
    uint32_t n = 0;
    RGB mean{0};
    RGB m2{0};
    RGB m3{0};

    // Preserve the first rejected sample for a host-side diagnostic rather
    // than asserting inside a rendering kernel and losing the CUDA context.
    uint32_t invalid_sample_n = 0;
    RGB invalid_sample{0};
    enum class Error : uint8_t { None, InvalidSample, MomentOverflow, CountOverflow };
    Error error = Error::None;

    // RGB is only a three-component container here: the observation is a
    // complete, unweighted sensor XYZ camera sample, including zero.
    // Multiplying by a signed reconstruction weight
    // before this call is unsupported. Calls for a pixel must be serialized.
    PBRT_CPU_GPU
    void add(const RGB &xyz) {
        if (n == UINT32_MAX) {
            record_error(xyz, Error::CountOverflow);
            return;
        }
        ++n;
        if (error != Error::None) {
            return;
        }
        if (xyz.has_nan() || xyz.r < 0 || xyz.g < 0 || xyz.b < 0) {
            record_error(xyz, Error::InvalidSample);
            return;
        }
        // Fixed Box-Cox lambda = 1/2. Invalid XYZ values are never clamped.
        const RGB transformed(2 * (std::sqrt(xyz.r) - 1),
                              2 * (std::sqrt(xyz.g) - 1),
                              2 * (std::sqrt(xyz.b) - 1));
        if (n == 1) {
            mean = transformed;
            return;
        }
        const RGB delta = transformed - mean;
        const RGB delta_n = delta / Real(n);
        mean += delta_n;
        m2 += delta * (delta - delta_n);
        // Meng's update uses the newly updated M2, as in paper Algorithm 1.
        m3 += delta * (delta * delta - delta_n * delta_n) - 3 * delta_n * m2;
        if (mean.has_nan() || m2.has_nan() || m3.has_nan()) {
            record_error(xyz, Error::MomentOverflow);
        }
    }

    // Paper Algorithm 2 / the reference johnson_mean_corrs_kernel: M3/n is
    // the third central moment; M2/(n-1) is the sample variance.
    PBRT_CPU_GPU
    bool estimate(RGB &corrected_mean, RGB &mean_variance) const {
        if (n < 2 || error != Error::None || mean.has_nan() || m2.has_nan() || m3.has_nan()) {
            return false;
        }
        corrected_mean = mean;
        for (uint8_t c = 0; c < 3; ++c) {
            if (m2[c] < 0) {
                return false;
            }
            const Real variance = m2[c] / Real(n - 1);
            mean_variance[c] = variance / Real(n);
            if (variance > std::numeric_limits<Real>::epsilon()) {
                corrected_mean[c] += (m3[c] / Real(n)) / variance / Real(n) / 6;
            }
        }
        return !corrected_mean.has_nan() && !mean_variance.has_nan();
    }

  private:
    PBRT_CPU_GPU
    void record_error(const RGB &sample, const Error reason) {
        if (error == Error::None) {
            invalid_sample_n = n;
            invalid_sample = sample;
            error = reason;
        }
    }
};

// The allocator owns the arrays and must outlive this object and its users.
class Denoiser {
  public:
    Denoiser(Point2i resolution, GPUMemoryAllocator &allocator);

    PixelStatistics *statistics() const {
        return pixel_statistics;
    }

    // Lazily allocate the full StatMC moments. Call before rendering and
    // attach the result to the producer of complete camera samples.
    TransformedStatistics *transformed_statistics();

    // The input must be GPU-accessible, and rendering/statistics updates must
    // be finished. Returns a synchronized, separately allocated managed image.
    const RGB *denoise(const RGB *linear_rgb);

    // Full mode uses Box-Cox / skewness-corrected XYZ moments, raw albedo and
    // normal guides, and pairwise linear luminance differences. Populate guide
    // luminance from this input image; the diagnostic luminance_difference
    // field is unused. Non-unit reconstruction filters use the unweighted
    // observation stream as an approximate gate for the weighted Film image.
    // Throws on any recorded negative, NaN, or infinite XYZ camera sample.
    const RGB *denoise(const RGB *linear_rgb, const GBufferPixel *guides);

    // Full-mode diagnostics, available after denoise: neighbors exclude the
    // center, while total weights include its fixed weight of one.
    const uint32_t *accepted_neighbors() const {
        return neighbor_counts;
    }

    const Real *weight_sums() const {
        return total_weights;
    }

  private:
    Point2i resolution;
    PixelStatistics *pixel_statistics = nullptr;
    RGB *output = nullptr;
    Real *critical_values = nullptr;
    GPUMemoryAllocator *allocator = nullptr;
    TransformedStatistics *transformed = nullptr;
    RGB *corrected_means = nullptr;
    RGB *mean_variances = nullptr;
    uint32_t *neighbor_counts = nullptr;
    Real *total_weights = nullptr;
};

} // namespace statmc

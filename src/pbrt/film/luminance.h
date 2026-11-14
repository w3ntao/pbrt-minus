#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <pbrt/base/film.h>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/film/heatmap.h>
#include <pbrt/spectrum_util/color_encoding.h>
#include <pbrt/spectrum_util/rgb.h>
#include <stdexcept>
#include <string>
#include <vector>

// Keep image-space luminance in the same linear XYZ Y units as the Film.
PBRT_CPU_GPU
inline Real linear_rgb_luminance(const RGB &rgb, const SquareMatrix<3> &xyz_from_rgb) {
    return rgb.has_nan() ? NAN : (xyz_from_rgb * rgb).g;
}

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
        if (n < 2 || mean.has_nan() || m2.has_nan() || covariance.has_nan() || !std::isfinite(mean_weight) ||
            !std::isfinite(m2_weight) || m2_weight < 0 || mean_weight == 0) {
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
        const RGB transformed(2 * (std::sqrt(xyz.r) - 1), 2 * (std::sqrt(xyz.g) - 1), 2 * (std::sqrt(xyz.b) - 1));
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

    // Apply the same radiance factor to all observations without replaying
    // samples: T(scale * XYZ) = sqrt(scale) * (T(XYZ) + 2) - 2.
    // This keeps the statistical guide consistent with a scaled beauty pixel.
    PBRT_CPU_GPU
    void rescale(const Real scale) {
        if (n == 0 || error != Error::None || scale == 1) {
            return;
        }
        if (!std::isfinite(scale) || scale < 0) {
            record_error(RGB(scale), Error::InvalidSample);
            return;
        }
        const Real root = std::sqrt(scale);
        mean = root * (mean + RGB(2)) - RGB(2);
        m2 = m2 * scale;
        m3 = m3 * (scale * root);
        if (mean.has_nan() || m2.has_nan() || m3.has_nan()) {
            record_error(RGB(NAN), Error::MomentOverflow);
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
    // normal guides, and pairwise linear luminance differences. The caller
    // supplies one linear XYZ Y value per input RGB pixel. Non-unit
    // reconstruction filters use the unweighted observation stream as an
    // approximate gate for the weighted Film image.
    // The surface guides, luminance, and input image remain unchanged.
    // Throws on any recorded negative, NaN, or infinite XYZ camera sample.
    const RGB *denoise(const RGB *linear_rgb, const GBufferPixel *guides, const Real *luminance);

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
    unsigned int *first_error_index = nullptr;
    uint32_t *neighbor_counts = nullptr;
    Real *total_weights = nullptr;
};

} // namespace statmc

namespace luminance_detail {

inline Real luminance_difference(const Real *luminance, const Point2i resolution, const int x, const int y) {
    const Real center = luminance[size_t(y) * resolution.x + x];
    if (!std::isfinite(center)) {
        return NAN;
    }

    Real max_difference = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= resolution.y) {
            continue;
        }
        for (int dx = -1; dx <= 1; ++dx) {
            const int nx = x + dx;
            if (nx < 0 || nx >= resolution.x) {
                continue;
            }
            const Real neighbor = luminance[size_t(ny) * resolution.x + nx];
            if (!std::isfinite(neighbor)) {
                return NAN;
            }
            max_difference = std::max(max_difference, std::abs(center - neighbor));
        }
    }
    return max_difference;
}

inline RGB viridis_rgb(const double value, const SRGBColorEncoding &srgb_encoding) {
    if (!std::isfinite(value)) {
        return RGB(0);
    }

    const auto &colors = heatmap::viridis_srgb;
    const double position = std::clamp(value, 0.0, 1.0) * (colors.size() - 1);
    const size_t index = std::min(size_t(position), colors.size() - 2);
    const Real t = Real(position - index);
    RGB rgb;
    for (int c = 0; c < 3; ++c) {
        // Interpolate in linear RGB; Film applies the shared sRGB PNG encoding.
        rgb[c] =
            (1 - t) * srgb_encoding.to_linear(colors[index][c]) + t * srgb_encoding.to_linear(colors[index + 1][c]);
    }
    return rgb;
}

inline void write_visualization(const std::string &filename, const Point2i resolution, const Real *values) {
    const size_t num_pixels = size_t(resolution.x) * resolution.y;
    std::vector<double> positive_values;
    positive_values.reserve(num_pixels);
    for (size_t i = 0; i < num_pixels; ++i) {
        const double value = double(values[i]);
        if (std::isfinite(value) && value > 0) {
            positive_values.push_back(value);
        }
    }

    double reference = 0;
    if (!positive_values.empty()) {
        // Keep sparse bright pixels from hiding weaker edges in the display.
        const size_t percentile = size_t(0.99 * double(positive_values.size() - 1));
        std::nth_element(positive_values.begin(), positive_values.begin() + percentile, positive_values.end());
        reference = positive_values[percentile];
    }

    const SRGBColorEncoding srgb_encoding;
    std::vector<RGB> rgb_pixels(num_pixels);
    for (size_t i = 0; i < num_pixels; ++i) {
        const double raw = double(values[i]);
        if (std::isfinite(raw) && raw > 0 && reference > 0) {
            constexpr double compression = 100;
            const double value = std::log1p(compression * raw / reference) / std::log1p(compression);
            rgb_pixels[i] = viridis_rgb(value, srgb_encoding);
        }
    }
    Film::write_linear_rgb_to_png(filename, resolution, rgb_pixels.data());
}

} // namespace luminance_detail

// GPU writes must be complete, and luminance must contain one CPU-readable
// linear CIE Y estimate per pixel. Writes luminance and maximum absolute 3x3
// luminance-difference PNGs beside output_filename without changing the input.
inline void write_luminance_diagnostics(const std::string &output_filename, const Point2i resolution,
                                        const Real *luminance) {
    if (resolution.x <= 0 || resolution.y <= 0 || luminance == nullptr) {
        throw std::invalid_argument("Luminance diagnostics require a positive resolution and a luminance image");
    }

    const std::filesystem::path path(output_filename);
    const auto prefix = path.parent_path() / path.stem();
    luminance_detail::write_visualization(prefix.string() + "-luminance.png", resolution, luminance);

    const size_t num_pixels = size_t(resolution.x) * resolution.y;
    std::vector<Real> differences(num_pixels);
    for (int y = 0; y < resolution.y; ++y) {
        for (int x = 0; x < resolution.x; ++x) {
            differences[size_t(y) * resolution.x + x] =
                luminance_detail::luminance_difference(luminance, resolution, x, y);
        }
    }
    luminance_detail::write_visualization(prefix.string() + "-luminance-difference.png", resolution,
                                          differences.data());
}

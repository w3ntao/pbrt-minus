#include <cmath>
#include <limits>
#include <pbrt/film/gbuffer.h>
#include <pbrt/film/statistical_denoiser.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <sstream>
#include <stdexcept>

namespace statmc {
namespace {

constexpr int max_degrees_of_freedom = 4096;
constexpr int filter_radius = 3;
constexpr Real filter_sigma = 3;
constexpr double significance = 0.005;
constexpr int threads = 256;

// The authors' scenes/denoise.pbrt specifies these standard deviations;
// each is squared in the Gaussian, rather than used as a covariance entry.
constexpr int statmc_radius = 20;
constexpr int statmc_max_degrees_of_freedom = 1024;
constexpr Real statmc_spatial_sd = 10;
constexpr Real statmc_albedo_sd = Real(0.02);
constexpr Real statmc_normal_sd = Real(0.1);

/*
statmc_luminance_relative_sd:
    additional a-trous-inspired luminance guide,
    expressed as a fraction of the brighter magnitude in each pair.
  original value: 0.2
*/
constexpr Real statmc_luminance_relative_sd = Real(0.4);

// Modified Lentz continued fraction for the regularized incomplete beta.
// Only construction uses this host-side calculation, never the pixel kernel.
double beta_fraction(const double a, const double b, const double x) {
    constexpr double tiny = 1e-300;
    constexpr double tolerance = 1e-14;
    const double qab = a + b, qap = a + 1, qam = a - 1;
    double c = 1;
    double d = 1 - qab * x / qap;
    if (std::abs(d) < tiny) {
        d = tiny;
    }
    d = 1 / d;
    double h = d;
    for (int m = 1; m <= 200; ++m) {
        const int m2 = 2 * m;
        double aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1 + aa * d;
        if (std::abs(d) < tiny) {
            d = tiny;
        }
        c = 1 + aa / c;
        if (std::abs(c) < tiny) {
            c = tiny;
        }
        d = 1 / d;
        h *= d * c;
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1 + aa * d;
        if (std::abs(d) < tiny) {
            d = tiny;
        }
        c = 1 + aa / c;
        if (std::abs(c) < tiny) {
            c = tiny;
        }
        d = 1 / d;
        const double step = d * c;
        h *= step;
        if (std::abs(step - 1) < tolerance) {
            return h;
        }
    }
    throw std::runtime_error("StatMC: incomplete beta did not converge");
}

double regularized_beta(const double a, const double b, const double x) {
    if (x <= 0) {
        return 0;
    }
    if (x >= 1) {
        return 1;
    }
    const double factor =
        std::exp(std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b) + a * std::log(x) + b * std::log1p(-x));
    if (x < (a + 1) / (a + b + 2)) {
        return factor * beta_fraction(a, b, x) / a;
    }
    return 1 - factor * beta_fraction(b, a, 1 - x) / b;
}

double critical_value(const int degrees_of_freedom) {
    const double nu = degrees_of_freedom;
    auto two_sided_tail = [nu](const double t) { return regularized_beta(nu / 2, 0.5, nu / (nu + t * t)); };
    double lower = 0, upper = 8;
    while (two_sided_tail(upper) > significance) {
        upper *= 2;
    }
    for (int iteration = 0; iteration < 60; ++iteration) {
        const double middle = (lower + upper) / 2;
        if (two_sided_tail(middle) > significance) {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    return upper;
}

PBRT_GPU
bool compatible(const RGB &mean_a, const RGB &variance_a, const uint32_t n_a, const RGB &mean_b, const RGB &variance_b,
                const uint32_t n_b, const Real *critical) {
    for (uint8_t c = 0; c < 3; ++c) {
        const Real va = variance_a[c];
        const Real vb = variance_b[c];
        const Real variance = va + vb;
        if (!std::isfinite(variance)) {
            return false;
        }
        if (variance == 0) {
            if (mean_a[c] != mean_b[c]) {
                return false;
            }
            continue;
        }
        // Normalize first to avoid squaring large or tiny variances in the
        // Welch-Satterthwaite degrees-of-freedom calculation.
        const Real fraction_a = va / variance;
        const Real fraction_b = vb / variance;
        const Real nu = 1 / (fraction_a * fraction_a / Real(n_a - 1) + fraction_b * fraction_b / Real(n_b - 1));
        // Flooring/capping gives a larger critical value: conservative for
        // rejecting equal means, rather than a guarantee against blurring.
        const int index = nu >= max_degrees_of_freedom ? max_degrees_of_freedom : (nu < 1 ? 1 : int(nu));
        const Real t = std::abs(mean_a[c] - mean_b[c]) / std::sqrt(variance);
        if (!(t < critical[index])) {
            return false;
        }
    }
    return true;
}

__global__ void initialize(PixelStatistics *statistics, RGB *output, const int count) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < count) {
        statistics[index] = {};
        output[index] = RGB(0);
    }
}

__global__ void filter(const RGB *input, RGB *output, const PixelStatistics *statistics, const Point2i resolution,
                       const Real *critical) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= resolution.x * resolution.y) {
        return;
    }
    const RGB center = input[index];
    output[index] = center;
    RGB center_mean, center_variance;
    if (center.has_nan() || !statistics[index].estimate(center_mean, center_variance)) {
        return;
    }
    const int x = index % resolution.x, y = index / resolution.x;
    RGB color = center;
    Real weight_sum = 1;
    for (int dy = -filter_radius; dy <= filter_radius; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= resolution.y) {
            continue;
        }
        for (int dx = -filter_radius; dx <= filter_radius; ++dx) {
            const int nx = x + dx;
            if (nx < 0 || nx >= resolution.x || (dx == 0 && dy == 0)) {
                continue;
            }
            const int neighbor = ny * resolution.x + nx;
            RGB neighbor_mean, neighbor_variance;
            if (input[neighbor].has_nan() || !statistics[neighbor].estimate(neighbor_mean, neighbor_variance) ||
                !compatible(center_mean, center_variance, statistics[index].n, neighbor_mean, neighbor_variance,
                            statistics[neighbor].n, critical)) {
                continue;
            }
            const Real weight = std::exp(-Real(dx * dx + dy * dy) / (2 * filter_sigma * filter_sigma));
            weight_sum += weight;
            const Real fraction = weight / weight_sum;
            // An incremental convex average avoids overflowing an unnormalized
            // RGB sum even when individual input colors are large but finite.
            color = color * (1 - fraction) + input[neighbor] * fraction;
        }
    }
    output[index] = color.has_nan() ? center : color;
}

__global__ void initialize_transformed(TransformedStatistics *statistics, uint32_t *neighbors, Real *weights,
                                       const int count) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < count) {
        statistics[index] = {};
        neighbors[index] = 0;
        weights[index] = 1;
    }
}

__global__ void prepare_transformed(const TransformedStatistics *statistics, RGB *means, RGB *variances,
                                    const int count) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < count && !statistics[index].estimate(means[index], variances[index])) {
        means[index] = variances[index] = RGB(NAN);
    }
}

PBRT_GPU
bool valid_guide(const GBufferPixel &guide) {
    if (!std::isfinite(guide.luminance) || !std::isfinite(guide.coverage) || guide.coverage < 0 || guide.coverage > 1) {
        return false;
    }
    if (guide.coverage == 0) {
        return true;
    }
    return !guide.albedo.has_nan() && !guide.normal.has_nan() &&
           (guide.normal.x != 0 || guide.normal.y != 0 || guide.normal.z != 0);
}

PBRT_GPU
Real guide_distance(const GBufferPixel &a, const GBufferPixel &b) {
    // Coverage is only a validity / surface-background boundary. Partial
    // coverage uses the existing conditional-on-hit attribute means, without
    // introducing a coverage-distance parameter absent from the paper.
    if ((a.coverage == 0) != (b.coverage == 0)) {
        return Infinity;
    }
    // Compare the raw linear Y values, not the diagnostic 3x3 edge image.
    // Normalize before subtraction to avoid overflow for finite HDR values;
    // the zero case also handles entirely black images without an epsilon
    // tied to exposure. Signed reconstruction-filter results remain valid.
    const Real luminance_scale = std::max(std::abs(a.luminance), std::abs(b.luminance));
    const Real luminance_delta =
        luminance_scale == 0
            ? 0
            : (a.luminance / luminance_scale - b.luminance / luminance_scale) / statmc_luminance_relative_sd;
    const Real luminance_d2 = luminance_delta * luminance_delta;
    if (a.coverage == 0) {
        return luminance_d2;
    }
    const RGB albedo = (a.albedo - b.albedo) / statmc_albedo_sd;
    // Preserve raw averaged normals; renormalizing would change the guides.
    const Vector3f normal = (a.normal - b.normal) / statmc_normal_sd;
    return luminance_d2 + albedo.r * albedo.r + albedo.g * albedo.g + albedo.b * albedo.b + normal.squared_length();
}

PBRT_GPU
bool transformed_compatible(const RGB &mean_a, const RGB &variance_a, const uint32_t n_a, const RGB &mean_b,
                            const RGB &variance_b, const uint32_t n_b, const Real *critical) {
    // Paper / reference implementation uses the upper-bound df. Use a wide
    // addition because the counters themselves are 32-bit.
    const uint64_t nu = uint64_t(n_a) + uint64_t(n_b) - 2;
    const int index = nu > statmc_max_degrees_of_freedom ? statmc_max_degrees_of_freedom : int(nu);
    for (uint8_t c = 0; c < 3; ++c) {
        const Real variance = variance_a[c] + variance_b[c];
        if (!std::isfinite(variance)) {
            return false;
        }
        if (variance == 0) {
            if (mean_a[c] != mean_b[c]) {
                return false;
            }
        } else if (!(std::abs(mean_a[c] - mean_b[c]) / std::sqrt(variance) < critical[index])) {
            return false;
        }
    }
    return true;
}

__global__ void filter_transformed(const RGB *input, RGB *output, const TransformedStatistics *statistics,
                                   const RGB *means, const RGB *variances, const GBufferPixel *guides,
                                   const Point2i resolution, const Real *critical, uint32_t *accepted_neighbors,
                                   Real *weight_sums) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= resolution.x * resolution.y) {
        return;
    }
    const RGB center = input[index];
    output[index] = center;
    accepted_neighbors[index] = 0;
    weight_sums[index] = 1;
    if (center.has_nan() || means[index].has_nan() || !valid_guide(guides[index])) {
        return;
    }
    const int x = index % resolution.x, y = index / resolution.x;
    RGB color = center;
    Real weight_sum = 1;
    uint32_t neighbor_count = 0;
    for (int dy = -statmc_radius; dy <= statmc_radius; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= resolution.y) {
            continue;
        }
        for (int dx = -statmc_radius; dx <= statmc_radius; ++dx) {
            const int nx = x + dx;
            const int distance_squared = dx * dx + dy * dy;
            if (nx < 0 || nx >= resolution.x || distance_squared > statmc_radius * statmc_radius ||
                (dx == 0 && dy == 0)) {
                continue;
            }
            const int neighbor = ny * resolution.x + nx;
            if (input[neighbor].has_nan() || means[neighbor].has_nan() || !valid_guide(guides[neighbor])) {
                continue;
            }
            const Real guide_d2 = guide_distance(guides[index], guides[neighbor]);
            if (!std::isfinite(guide_d2) ||
                !transformed_compatible(means[index], variances[index], statistics[index].n, means[neighbor],
                                        variances[neighbor], statistics[neighbor].n, critical)) {
                continue;
            }
            const Real weight =
                std::exp(-Real(0.5) * (Real(distance_squared) / (statmc_spatial_sd * statmc_spatial_sd) + guide_d2));
            if (weight == 0) {
                continue;
            }
            ++neighbor_count;
            weight_sum += weight;
            const Real fraction = weight / weight_sum;
            color = color * (1 - fraction) + input[neighbor] * fraction;
        }
    }
    if (!color.has_nan()) {
        output[index] = color;
        accepted_neighbors[index] = neighbor_count;
        weight_sums[index] = weight_sum;
    }
}

} // namespace

Denoiser::Denoiser(const Point2i resolution, GPUMemoryAllocator &allocator)
    : resolution(resolution), allocator(&allocator) {
    if (resolution.x <= 0 || resolution.y <= 0 ||
        size_t(resolution.x) * size_t(resolution.y) > size_t(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("StatMC: invalid image resolution");
    }
    const int count = resolution.x * resolution.y;
    pixel_statistics = allocator.allocate<PixelStatistics>(count);
    output = allocator.allocate<RGB>(count);
    critical_values = allocator.allocate<Real>(max_degrees_of_freedom + 1);
    critical_values[0] = NAN;
    for (int nu = 1; nu <= max_degrees_of_freedom; ++nu) {
        critical_values[nu] = std::nextafter(Real(critical_value(nu)), std::numeric_limits<Real>::infinity());
    }
    initialize<<<divide_and_ceil(count, threads), threads>>>(pixel_statistics, output, count);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

TransformedStatistics *Denoiser::transformed_statistics() {
    if (transformed == nullptr) {
        const int count = resolution.x * resolution.y;
        transformed = allocator->allocate<TransformedStatistics>(count);
        corrected_means = allocator->allocate<RGB>(count);
        mean_variances = allocator->allocate<RGB>(count);
        neighbor_counts = allocator->allocate<uint32_t>(count);
        total_weights = allocator->allocate<Real>(count);
        initialize_transformed<<<divide_and_ceil(count, threads), threads>>>(transformed, neighbor_counts,
                                                                             total_weights, count);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    return transformed;
}

const RGB *Denoiser::denoise(const RGB *linear_rgb) {
    if (linear_rgb == nullptr || linear_rgb == output) {
        throw std::invalid_argument("StatMC: expected a separate input image");
    }
    const int count = resolution.x * resolution.y;
    filter<<<divide_and_ceil(count, threads), threads>>>(linear_rgb, output, pixel_statistics, resolution,
                                                         critical_values);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return output;
}

const RGB *Denoiser::denoise(const RGB *linear_rgb, const GBufferPixel *guides) {
    if (linear_rgb == nullptr || linear_rgb == output || guides == nullptr) {
        throw std::invalid_argument("StatMC: expected a separate input image and raw G-buffer guides");
    }
    if (transformed == nullptr) {
        throw std::logic_error("StatMC: transformed statistics must be attached before rendering");
    }
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    const int count = resolution.x * resolution.y;
    for (int index = 0; index < count; ++index) {
        const auto &s = transformed[index];
        if (s.error == TransformedStatistics::Error::None) {
            continue;
        }
        std::ostringstream message;
        message.precision(std::numeric_limits<Real>::max_digits10);
        message << "StatMC: pixel (" << index % resolution.x << ", " << index / resolution.x << "), sample "
                << s.invalid_sample_n << ": ";
        if (s.error == TransformedStatistics::Error::InvalidSample) {
            message << "Box-Cox requires finite nonnegative sensor XYZ; received ";
        } else if (s.error == TransformedStatistics::Error::MomentOverflow) {
            message << "Box-Cox moment accumulation overflowed for ";
        } else {
            message << "camera-sample counter overflowed for ";
        }
        message << "XYZ(" << s.invalid_sample.r << ", " << s.invalid_sample.g << ", " << s.invalid_sample.b
                << "); samples observed: " << s.n;
        throw std::runtime_error(message.str());
    }
    prepare_transformed<<<divide_and_ceil(count, threads), threads>>>(transformed, corrected_means, mean_variances,
                                                                      count);
    CHECK_CUDA_ERROR(cudaGetLastError());
    filter_transformed<<<divide_and_ceil(count, threads), threads>>>(linear_rgb, output, transformed, corrected_means,
                                                                     mean_variances, guides, resolution,
                                                                     critical_values, neighbor_counts, total_weights);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return output;
}

} // namespace statmc

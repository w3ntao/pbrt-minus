#include <cmath>
#include <limits>
#include <pbrt/film/statistical_denoiser.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <stdexcept>

namespace statmc {
namespace {

constexpr int max_degrees_of_freedom = 4096;
constexpr int filter_radius = 3;
constexpr Real filter_sigma = 3;
constexpr double significance = 0.005;
constexpr int threads = 256;

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

} // namespace

Denoiser::Denoiser(const Point2i resolution, GPUMemoryAllocator &allocator) : resolution(resolution) {
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
    initialize<<<(count + threads - 1) / threads, threads>>>(pixel_statistics, output, count);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

const RGB *Denoiser::denoise(const RGB *linear_rgb) {
    if (linear_rgb == nullptr || linear_rgb == output) {
        throw std::invalid_argument("StatMC: expected a separate input image");
    }
    const int count = resolution.x * resolution.y;
    filter<<<(count + threads - 1) / threads, threads>>>(linear_rgb, output, pixel_statistics, resolution,
                                                         critical_values);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return output;
}

} // namespace statmc

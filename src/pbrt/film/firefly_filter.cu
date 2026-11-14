#include <algorithm>
#include <cmath>
#include <limits>
#include <pbrt/film/firefly_filter.h>
#include <pbrt/film/gbuffer.h>
#include <pbrt/film/luminance.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <stdexcept>

namespace firefly {
namespace {

constexpr int radius = 2;
constexpr int max_neighbors = (2 * radius + 1) * (2 * radius + 1) - 1;
constexpr int min_neighbors = 8;
constexpr int threads = 256;
constexpr Real max_coverage_difference = Real(0.25);
constexpr Real max_normal_distance_squared = Real(0.2) * Real(0.2);
constexpr Real max_albedo_distance_squared = Real(0.1) * Real(0.1);

PBRT_GPU
bool valid_guide(const GBufferPixel &guide, const Real luminance) {
    if (!std::isfinite(luminance) || luminance < 0 || !std::isfinite(guide.coverage) || guide.coverage < 0 ||
        guide.coverage > 1) {
        return false;
    }
    if (guide.coverage == 0) {
        return true;
    }
    const Real normal_length_squared = guide.normal.squared_length();
    return !guide.albedo.has_nan() && !guide.normal.has_nan() && std::isfinite(normal_length_squared) &&
           normal_length_squared > 0;
}

PBRT_GPU
bool compatible_guides(const GBufferPixel &center, const GBufferPixel &neighbor) {
    // Do not borrow a dark surface/background across a geometric boundary.
    if ((center.coverage == 0) != (neighbor.coverage == 0) ||
        std::abs(center.coverage - neighbor.coverage) > max_coverage_difference) {
        return false;
    }
    if (center.coverage == 0) {
        return true;
    }
    const auto normal_delta = center.normal - neighbor.normal;
    const auto albedo_delta = center.albedo - neighbor.albedo;
    return normal_delta.squared_length() <= max_normal_distance_squared &&
           albedo_delta.r * albedo_delta.r + albedo_delta.g * albedo_delta.g + albedo_delta.b * albedo_delta.b <=
               max_albedo_distance_squared;
}

PBRT_GPU
void sort(Real *values, const int count) {
    // At most 24 entries per thread; no global sorting or extra launches.
    for (int i = 1; i < count; ++i) {
        const Real value = values[i];
        int j = i;
        while (j > 0 && values[j - 1] > value) {
            values[j] = values[j - 1];
            --j;
        }
        values[j] = value;
    }
}

PBRT_GPU
Real median(const Real *sorted_values, const int count) {
    return count % 2 ? sorted_values[count / 2]
                     : Real(0.5) * sorted_values[count / 2 - 1] + Real(0.5) * sorted_values[count / 2];
}

__global__ void remove_isolated_fireflies(const RGB *input, RGB *output, Real *scales, const GBufferPixel *guides,
                                          const Real *luminance, statmc::TransformedStatistics *statistics,
                                          const Point2i resolution, unsigned int *modified_pixel_count) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= resolution.x * resolution.y) {
        return;
    }
    const RGB center_rgb = input[index];
    const auto &center_guide = guides[index];
    const Real center_luminance = luminance[index];
    output[index] = center_rgb;
    scales[index] = 1;
    if (center_rgb.has_nan() || !valid_guide(center_guide, center_luminance) || !(center_luminance > 0)) {
        return;
    }
    if (statistics != nullptr) {
        const auto &s = statistics[index];
        // Identical complete camera observations support a real tiny bright
        // feature even without bright spatial neighbors. Do not override an
        // error or infer stability from an unpopulated/one-sample guide.
        if (s.n >= 2 && s.error == statmc::TransformedStatistics::Error::None && !s.mean.has_nan() && s.m2 == RGB(0) &&
            s.m3 == RGB(0)) {
            return;
        }
    }

    Real luminances[max_neighbors];
    int count = 0;
    Real neighborhood_scale = 0;
    const int x = index % resolution.x, y = index / resolution.x;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int ny = y + dy;
        if (ny < 0 || ny >= resolution.y) {
            continue;
        }
        for (int dx = -radius; dx <= radius; ++dx) {
            const int nx = x + dx;
            if (nx < 0 || nx >= resolution.x || (dx == 0 && dy == 0)) {
                continue;
            }
            const int neighbor = ny * resolution.x + nx;
            const auto &guide = guides[neighbor];
            const Real neighbor_luminance = luminance[neighbor];
            if (input[neighbor].has_nan() || !valid_guide(guide, neighbor_luminance) ||
                !compatible_guides(center_guide, guide)) {
                continue;
            }
            luminances[count++] = neighbor_luminance;
            neighborhood_scale = std::max(neighborhood_scale, neighbor_luminance);
        }
    }
    if (count < min_neighbors) {
        return;
    }

    Real scale = 0;
    if (neighborhood_scale > 0) {
        // Normalize first to keep median/MAD and thresholds finite for HDR
        // input, with no exposure-dependent epsilon or global ceiling.
        for (int i = 0; i < count; ++i) {
            luminances[i] /= neighborhood_scale;
        }
        sort(luminances, count);
        const Real local_median = median(luminances, count);
        // A supported bright tail (e.g. a highlight patch or thin line) must
        // raise the limit even when a dark majority makes the MAD vanish.
        const Real upper_tail = luminances[(9 * count - 1) / 10];
        for (int i = 0; i < count; ++i) {
            luminances[i] = std::abs(luminances[i] - local_median);
        }
        sort(luminances, count);
        const Real mad = median(luminances, count);
        const Real robust_sigma = Real(1.4826) * mad;
        const Real detection_limit =
            std::max(std::max(8 * local_median, local_median + 8 * robust_sigma), 2 * upper_tail);
        // Once the conservative test identifies an isolated outlier, bring
        // it into the supported local tail rather than leaving it at the
        // much higher detection threshold (which can remain visibly bright).
        const Real clamp_limit = std::max(upper_tail, local_median + 3 * robust_sigma);

        // Compare and divide in this order to avoid overflowing either an
        // absolute HDR limit or a ratio between unrelated exposure scales.
        scale = 1;
        if (center_luminance <= neighborhood_scale) {
            const Real normalized_center = center_luminance / neighborhood_scale;
            if (normalized_center > detection_limit) {
                scale = clamp_limit / normalized_center;
            }
        } else {
            const Real relative_neighborhood = neighborhood_scale / center_luminance;
            if (detection_limit * relative_neighborhood < 1) {
                scale = clamp_limit * relative_neighborhood;
            }
        }
    }

    if (scale < 1) {
        // A common scale preserves chromaticity and signed RGB channels.
        output[index] = center_rgb * scale;
        scales[index] = scale;
        atomicAdd(modified_pixel_count, 1u);
        if (statistics != nullptr) {
            // This maps the original observation stream to uniformly scaled
            // XYZ samples; it does not reconstruct unknown rejected samples.
            // rescale() preserves counts and recorded error diagnostics.
            statistics[index].rescale(scale);
        }
    }
}

__global__ void apply_pixel_scales(Real *luminance, const Real *scales, const int count) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < count) {
        luminance[index] *= scales[index];
    }
}

} // namespace

Prefilter::Prefilter(const Point2i resolution, GPUMemoryAllocator &allocator) : resolution(resolution) {
    if (resolution.x <= 0 || resolution.y <= 0 ||
        size_t(resolution.x) > size_t(std::numeric_limits<int>::max()) / size_t(resolution.y)) {
        throw std::invalid_argument("Firefly prefilter: invalid image resolution");
    }
    const size_t count = size_t(resolution.x) * size_t(resolution.y);
    output = allocator.allocate<RGB>(count);
    pixel_scales = allocator.allocate<Real>(count);
    modified_pixel_count_gpu = allocator.allocate<unsigned int>();
}

const RGB *Prefilter::filter(const RGB *linear_rgb, const GBufferPixel *guides, const Real *luminance,
                             statmc::TransformedStatistics *statistics) {
    if (linear_rgb == nullptr || linear_rgb == output || guides == nullptr || luminance == nullptr) {
        throw std::invalid_argument(
            "Firefly prefilter: expected a separate input image, surface guides, and luminance");
    }
    const int count = resolution.x * resolution.y;
    CHECK_CUDA_ERROR(cudaMemsetAsync(modified_pixel_count_gpu, 0, sizeof(unsigned int)));
    remove_isolated_fireflies<<<divide_and_ceil(count, threads), threads>>>(
        linear_rgb, output, pixel_scales, guides, luminance, statistics, resolution, modified_pixel_count_gpu);
    CHECK_CUDA_ERROR(cudaGetLastError());
    unsigned int modified_count = 0;
    // This readback also waits for the filter, without migrating the scale image.
    CHECK_CUDA_ERROR(
        cudaMemcpy(&modified_count, modified_pixel_count_gpu, sizeof(modified_count), cudaMemcpyDeviceToHost));
    modified_pixel_count = modified_count;
    return output;
}

void Prefilter::apply_scales_to_luminance(Real *luminance) const {
    if (luminance == nullptr) {
        throw std::invalid_argument("Firefly prefilter: expected a luminance guide to scale");
    }
    const int count = resolution.x * resolution.y;
    apply_pixel_scales<<<divide_and_ceil(count, threads), threads>>>(luminance, pixel_scales, count);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace firefly

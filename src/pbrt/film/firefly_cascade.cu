#include <filesystem>
#include <pbrt/film/firefly_cascade.h>
#include <pbrt/film/grey_scale_film.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/util/math.h>

namespace firefly {

static void write_firefly_cascade(Real *const *counts, const Point2i resolution, const std::string &output_filename) {
    const auto image_path = std::filesystem::path(output_filename);
    const int num_pixels = resolution.x * resolution.y;
    printf("cascaded frame mask - nonzero pixels:\n");
    for (int layer_index = 0; layer_index < layer_count; ++layer_index) {
        GreyScaleFilm layer(resolution);
        size_t nonzero_pixels = 0;
        for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
            // Bitterli's buffers store sample counts, rather than layer radiance.
            const Real count = counts[layer_index][pixel_index];
            layer.add_sample(Point2i(pixel_index % resolution.x, pixel_index / resolution.x), count);
            nonzero_pixels += count != 0;
        }

        const auto layer_path = image_path.parent_path() / (image_path.stem().string() + "-cascaded-frame-" +
                                                            std::to_string(layer_index) + ".png");
        layer.write_to_png(layer_path.string());
        printf("    level %d: %6.2f%%\n", layer_index, double(nonzero_pixels) / num_pixels * 100);
    }
}

__global__ static void init_firefly_ranges(WeightRange **ranges, const int num_pixels) {
    const int pixel_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (pixel_index >= num_pixels) {
        return;
    }
    for (int layer = 0; layer < layer_count - 1; ++layer) {
        ranges[layer][pixel_index] = {};
    }
}

__global__ static void detect_fireflies(const CascadeView detector, const Real threshold, const int num_pixels) {
    const int pixel_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (pixel_index >= num_pixels) {
        return;
    }

    detector.mask[pixel_index] =
        is_candidate(detector.counts, detector.ranges, pixel_index, threshold, detector.mask[pixel_index] != 0);
}

Cascade::Cascade(const Point2i resolution, GPUMemoryAllocator &allocator) : resolution(resolution) {
    const int num_pixels = resolution.x * resolution.y;
    // Fixed-size log-luminance counts and observed interpolation-weight ranges
    // allow final classification of all actual samples without retracing.
    detector.counts = allocator.allocate<Real *>(layer_count);
    detector.ranges = allocator.allocate<WeightRange *>(layer_count - 1);
    for (int layer = 0; layer < layer_count; ++layer) {
        detector.counts[layer] = allocator.allocate<Real>(num_pixels);
        CHECK_CUDA_ERROR(cudaMemset(detector.counts[layer], 0, sizeof(Real) * num_pixels));
        if (layer + 1 < layer_count) {
            detector.ranges[layer] = allocator.allocate<WeightRange>(num_pixels);
        }
    }
    detector.mask = allocator.allocate<uint8_t>(num_pixels);
    CHECK_CUDA_ERROR(cudaMemset(detector.mask, 0, sizeof(uint8_t) * num_pixels));
    constexpr int threads = 256;
    init_firefly_ranges<<<divide_and_ceil(num_pixels, threads), threads>>>(detector.ranges, num_pixels);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void Cascade::finalize(GreyScaleFilm &firefly_mask, const int samples_per_pixel,
                       const std::string &output_filename) const {
    const int num_pixels = resolution.x * resolution.y;
    // Classification sees a read-only snapshot containing all rendered spp.
    const Real threshold = tau(samples_per_pixel);
    constexpr int threads = 256;
    detect_fireflies<<<divide_and_ceil(num_pixels, threads), threads>>>(detector, threshold, num_pixels);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    size_t candidate_pixels = 0;
    for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
        const auto value = detector.mask[pixel_index];
        firefly_mask.add_sample(Point2i(pixel_index % resolution.x, pixel_index / resolution.x), value);
        candidate_pixels += value;
    }
    printf("firefly detector: %d layers (alpha %.3g, beta %d, tau %.6g, max %.1f), candidate: %.1f%%\n", layer_count,
           double(alpha), beta, double(threshold), double(max_luminance), float(candidate_pixels) / num_pixels * 100);

    // Export all count layers, including empty ones, before the allocator releases them.
    write_firefly_cascade(detector.counts, resolution, output_filename);
}

} // namespace firefly

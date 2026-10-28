#include <ext/lodepng/lodepng.h>
#include <pbrt/base/film.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/spectrum_util/color_encoding.h>
#include <vector>

Film *Film::create_rgb_film(const Filter *filter, const ParameterDictionary &parameters,
                            GPUMemoryAllocator &allocator) {
    return allocator.create<Film>(RGBFilm(filter, parameters, allocator));
}

Film *Film::create_empty(GPUMemoryAllocator &allocator) const {
    return cuda::std::visit([&](const auto &x) { return allocator.create<Film>(x.create_empty(allocator)); }, *this);
}

void Film::attach_statistics(statmc::PixelStatistics *statistics) {
    cuda::std::visit([&](auto &x) { x.attach_statistics(statistics); }, *this);
}

void Film::add_splats(const Film &other, Real scale) {
    cuda::std::visit([&](auto &x, const auto &y) { x.add_splats(y, scale); }, *this, other);
}

void Film::merge(const Film &other) {
    cuda::std::visit([&](auto &x, const auto &y) { x.merge(y); }, *this, other);
}

__global__ void copy_pixels(uint8_t *gpu_frame_buffer, const Film *film, int width, int height, Real splat_scale) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const auto worker_idx = y * width + x;

    const auto rgb = film->get_pixel_rgb(Point2i(x, y), splat_scale);
    if (rgb.has_nan()) {
        gpu_frame_buffer[worker_idx * 3 + 0] = 0;
        gpu_frame_buffer[worker_idx * 3 + 1] = 0;
        gpu_frame_buffer[worker_idx * 3 + 2] = 0;

        return;
    }

    const SRGBColorEncoding srgb_encoding;

    gpu_frame_buffer[worker_idx * 3 + 0] = srgb_encoding.from_linear(rgb.r);
    gpu_frame_buffer[worker_idx * 3 + 1] = srgb_encoding.from_linear(rgb.g);
    gpu_frame_buffer[worker_idx * 3 + 2] = srgb_encoding.from_linear(rgb.b);
}

void Film::copy_to_frame_buffer(uint8_t *gpu_frame_buffer, Real splat_scale) const {
    const auto image_resolution = this->get_resolution();

    constexpr int thread_width = 16;
    constexpr int thread_height = 16;

    dim3 blocks(divide_and_ceil(int(image_resolution.x), thread_width),
                divide_and_ceil(int(image_resolution.y), thread_height));
    dim3 threads(thread_width, thread_height);

    copy_pixels<<<blocks, threads>>>(gpu_frame_buffer, this, image_resolution.x, image_resolution.y, splat_scale);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

static __global__ void copy_linear_pixels(RGB *pixels, const Film *film, int num_pixels,
                                           Real splat_scale) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_pixels) {
        return;
    }

    const auto resolution = film->get_resolution();
    pixels[index] = film->get_pixel_rgb(Point2i(index % resolution.x, index / resolution.x), splat_scale);
}

void Film::copy_to_linear_rgb(RGB *pixels, Real splat_scale) const {
    if (!pixels) {
        REPORT_FATAL_ERROR();
    }

    const auto resolution = get_resolution();
    const int num_pixels = resolution.x * resolution.y;
    constexpr int threads = 256;
    copy_linear_pixels<<<divide_and_ceil(num_pixels, threads), threads>>>(pixels, this, num_pixels, splat_scale);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

namespace {

template <typename ReadPixel>
void write_rgb_to_png(const std::string &filename, Point2i resolution, const char *operation,
                       ReadPixel read_pixel) {
    const int width = resolution.x;
    const int height = resolution.y;
    if (width <= 0 || height <= 0) {
        REPORT_FATAL_ERROR();
    }

    SRGBColorEncoding srgb_encoding;
    std::vector<unsigned char> png_pixels(width * height * 4);

    int invalid_pixels = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const int index = y * width + x;
            const auto rgb = read_pixel(Point2i(x, y));
            if (rgb.has_nan()) {
                printf("%sFilm::%s(): pixel (%d, %d) carries non-finite components%s\n", FLAG_COLORFUL_PRINT_RED_START,
                       operation, x, y, FLAG_COLORFUL_PRINT_END);
                invalid_pixels += 1;
                continue;
            }

            png_pixels[4 * index + 0] = srgb_encoding.from_linear(rgb.r);
            png_pixels[4 * index + 1] = srgb_encoding.from_linear(rgb.g);
            png_pixels[4 * index + 2] = srgb_encoding.from_linear(rgb.b);
            png_pixels[4 * index + 3] = 255;
        }
    }

    if (invalid_pixels > 0) {
        printf("%sFilm::%s(): %d/%d (%.1f%%) pixels carry non-finite components%s\n", FLAG_COLORFUL_PRINT_RED_START, operation,
               invalid_pixels, width * height, static_cast<Real>(invalid_pixels) / (width * height) * 100,
               FLAG_COLORFUL_PRINT_END);

        REPORT_FATAL_ERROR();
    }

    // Encode the image
    // if there's an error, display it
    if (unsigned error = lodepng::encode(filename, png_pixels, width, height); error) {
        std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        throw std::runtime_error("lodepng::encode() fail");
    }
}

} // namespace

void Film::write_linear_rgb_to_png(const std::string &filename, Point2i resolution, const RGB *pixels) {
    if (!pixels) {
        REPORT_FATAL_ERROR();
    }
    write_rgb_to_png(filename, resolution, __func__,
                     [&](const Point2i p) { return pixels[p.x + p.y * resolution.x]; });
}

void Film::write_to_png(const std::string &filename, Real splat_scale) const {
    write_rgb_to_png(filename, get_resolution(), __func__,
                     [&](const Point2i p) { return get_pixel_rgb(p, splat_scale); });
}

#include <ext/lodepng/lodepng.h>
#include <pbrt/base/film.h>
#include <pbrt/film/rgb_film.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/spectrum_util/color_encoding.h>
#include <vector>

Film *Film::create_rgb_film(const Filter *filter, const ParameterDictionary &parameters,
                            GPUMemoryAllocator &allocator) {
    auto rgb_film = RGBFilm::create(filter, parameters, allocator);

    auto film = allocator.allocate<Film>();
    film->init(rgb_film);

    return film;
}

void Film::init(RGBFilm *rgb_film) {
    ptr = rgb_film;
    type = Type::rgb;
}

PBRT_CPU_GPU
Point2i Film::get_resolution() const {
    switch (type) {
    case Type::rgb: {
        return static_cast<const RGBFilm *>(ptr)->get_resolution();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
const Filter *Film::get_filter() const {
    switch (type) {
    case Type::rgb: {
        return static_cast<const RGBFilm *>(ptr)->get_filter();
    }
    }

    REPORT_FATAL_ERROR();
    return nullptr;
}

PBRT_CPU_GPU
Bounds2f Film::sample_bounds() const {
    switch (type) {
    case Type::rgb: {
        return static_cast<const RGBFilm *>(ptr)->sample_bounds();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
void Film::add_sample(uint pixel_index, const SampledSpectrum &radiance_l,
                      const SampledWavelengths &lambda, Real weight) {
    switch (type) {
    case Type::rgb: {
        return static_cast<RGBFilm *>(ptr)->add_sample(pixel_index, radiance_l, lambda, weight);
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_CPU_GPU
void Film::add_sample(const Point2i &p_film, const SampledSpectrum &radiance_l,
                      const SampledWavelengths &lambda, Real weight) {
    switch (type) {
    case Type::rgb: {
        return static_cast<RGBFilm *>(ptr)->add_sample(p_film, radiance_l, lambda, weight);
    }
    }

    REPORT_FATAL_ERROR();
}

void Film::add_splat(const Point2f &p_film, const SampledSpectrum &radiance_l,
                     const SampledWavelengths &lambda) {
    switch (type) {
    case Type::rgb: {
        return static_cast<RGBFilm *>(ptr)->add_splat(p_film, radiance_l, lambda);
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_CPU_GPU
RGB Film::get_pixel_rgb(const Point2i &p, Real splat_scale) const {
    switch (type) {
    case Type::rgb: {
        return static_cast<RGBFilm *>(ptr)->get_pixel_rgb(p, splat_scale);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

__global__ void copy_pixels(uint8_t *gpu_frame_buffer, const Film *film, uint width, uint height,
                            Real splat_scale) {
    const auto x = blockIdx.x * blockDim.x + threadIdx.x;
    const auto y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const auto worker_idx = y * width + x;

    auto rgb = film->get_pixel_rgb(Point2i(x, y), splat_scale);
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

    constexpr uint thread_width = 16;
    constexpr uint thread_height = 16;

    dim3 blocks(divide_and_ceil(uint(image_resolution.x), thread_width),
                divide_and_ceil(uint(image_resolution.y), thread_height));
    dim3 threads(thread_width, thread_height);

    copy_pixels<<<blocks, threads>>>(gpu_frame_buffer, this, image_resolution.x, image_resolution.y,
                                     splat_scale);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void Film::write_to_png(const std::string &filename, Real splat_scale) const {
    auto resolution = get_resolution();

    int width = resolution.x;
    int height = resolution.y;

    SRGBColorEncoding srgb_encoding;
    std::vector<unsigned char> png_pixels(width * height * 4);

    uint nan_pixels = 0;

    for (uint y = 0; y < height; y++) {
        for (uint x = 0; x < width; x++) {
            uint index = y * width + x;
            auto rgb = get_pixel_rgb(Point2i(x, y), splat_scale);
            if (rgb.has_nan()) {
                nan_pixels += 1;
            }

            png_pixels[4 * index + 0] = srgb_encoding.from_linear(rgb.r);
            png_pixels[4 * index + 1] = srgb_encoding.from_linear(rgb.g);
            png_pixels[4 * index + 2] = srgb_encoding.from_linear(rgb.b);
            png_pixels[4 * index + 3] = 255;
        }
    }

    if (nan_pixels > 0) {
        printf("%sFilm::%s(): %d/%d (%.2f%) pixels with NAN component%s\n",
               FLAG_COLORFUL_PRINT_RED_START, __func__, nan_pixels, width * height,
               double(nan_pixels) / (width * height) * 100, FLAG_COLORFUL_PRINT_END);
    }

    // Encode the image
    // if there's an error, display it
    if (unsigned error = lodepng::encode(filename, png_pixels, width, height); error) {
        std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error)
                  << std::endl;
        throw std::runtime_error("lodepng::encode() fail");
    }
}

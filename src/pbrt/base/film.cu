#include "ext/lodepng/lodepng.h"
#include "pbrt/base/film.h"
#include "pbrt/films/rgb_film.h"
#include "pbrt/spectrum_util/color_encoding.h"
#include <vector>

Film *Film::create_rgb_film(const ParameterDictionary &parameters,
                            std::vector<void *> &gpu_dynamic_pointers) {
    Film *film;
    CHECK_CUDA_ERROR(cudaMallocManaged(&film, sizeof(Film)));
    gpu_dynamic_pointers.push_back(film);

    auto rgb_film = RGBFilm::create(parameters, gpu_dynamic_pointers);
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
    case (Type::rgb): {
        return ((RGBFilm *)ptr)->get_resolution();
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_CPU_GPU
void Film::add_sample(uint pixel_index, const SampledSpectrum &radiance_l,
                      const SampledWavelengths &lambda, FloatType weight) {
    switch (type) {
    case (Type::rgb): {
        return ((RGBFilm *)ptr)->add_sample(pixel_index, radiance_l, lambda, weight);
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_CPU_GPU
void Film::add_sample(const Point2i &p_film, const SampledSpectrum &radiance_l,
                      const SampledWavelengths &lambda, FloatType weight) {
    switch (type) {
    case (Type::rgb): {
        return ((RGBFilm *)ptr)->add_sample(p_film, radiance_l, lambda, weight);
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_CPU_GPU
RGB Film::get_pixel_rgb(const Point2i &p) const {
    switch (type) {
    case (Type::rgb): {
        return ((RGBFilm *)ptr)->get_pixel_rgb(p);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

void Film::write_to_png(const std::string &filename) const {
    auto resolution = get_resolution();

    int width = resolution.x;
    int height = resolution.y;

    SRGBColorEncoding srgb_encoding;
    std::vector<unsigned char> png_pixels(width * height * 4);

    uint nan_pixels = 0;

    for (uint y = 0; y < height; y++) {
        for (uint x = 0; x < width; x++) {
            uint index = y * width + x;
            auto rgb = get_pixel_rgb(Point2i(x, y));
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
        printf("%s(): %d/%d (roughly 1/%.1f) pixel has NAN component\n", __func__, nan_pixels,
               width * height, FloatType(width * height) / nan_pixels);
    }

    // Encode the image
    // if there's an error, display it
    if (unsigned error = lodepng::encode(filename, png_pixels, width, height); error) {
        std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error)
                  << std::endl;
        throw std::runtime_error("lodepng::encode() fail");
    }
}

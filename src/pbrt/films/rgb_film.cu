#include "pbrt/films/rgb_film.h"

#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/spectrum_util/global_spectra.h"

static __global__ void init_pixels(Pixel *pixels, Point2i dimension) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= dimension.x * dimension.y) {
        return;
    }

    pixels[idx].init_zero();
}

RGBFilm *RGBFilm::create(const ParameterDictionary &parameters,
                         std::vector<void *> &gpu_dynamic_pointers) {
    auto resolution_x = parameters.get_integer("xresolution")[0];
    auto resolution_y = parameters.get_integer("yresolution")[0];

    auto film_resolution = Point2i(resolution_x, resolution_y);

    FloatType iso = 100;
    FloatType white_balance_val = 0.0;
    FloatType exposure_time = 1.0;
    FloatType imaging_ratio = exposure_time * iso / 100.0;

    auto d_illum =
        Spectrum::create_cie_d(white_balance_val == 0.0 ? 6500.0 : white_balance_val, CIE_S0,
                               CIE_S1, CIE_S2, CIE_S_lambda, gpu_dynamic_pointers);

    PixelSensor *sensor;
    CHECK_CUDA_ERROR(cudaMallocManaged(&sensor, sizeof(PixelSensor)));
    gpu_dynamic_pointers.push_back(sensor);

    sensor->init_cie_1931(parameters.global_spectra->cie_xyz,
                          parameters.global_spectra->rgb_color_space,
                          white_balance_val == 0 ? nullptr : d_illum, imaging_ratio);

    Pixel *gpu_pixels;
    CHECK_CUDA_ERROR(
        cudaMallocManaged(&gpu_pixels, sizeof(Pixel) * film_resolution.x * film_resolution.y));
    gpu_dynamic_pointers.push_back(gpu_pixels);

    {
        uint threads = 1024;
        uint blocks = divide_and_ceil(uint(film_resolution.x * film_resolution.y), threads);

        init_pixels<<<blocks, threads>>>(gpu_pixels, film_resolution);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    RGBFilm *rgb_film;
    CHECK_CUDA_ERROR(cudaMallocManaged(&rgb_film, sizeof(RGBFilm)));
    gpu_dynamic_pointers.push_back(rgb_film);

    rgb_film->init(gpu_pixels, sensor, film_resolution, parameters.global_spectra->rgb_color_space);

    return rgb_film;
}

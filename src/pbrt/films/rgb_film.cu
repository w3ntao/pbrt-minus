#include <pbrt/base/filter.h>
#include <pbrt/films/pixel_sensor.h>
#include <pbrt/films/rgb_film.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/spectrum_util/rgb_color_space.h>

#include <pbrt/gpu/gpu_memory_allocator.h>

static __global__ void init_pixels(Pixel *pixels, Point2i dimension) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= dimension.x * dimension.y) {
        return;
    }

    pixels[idx].init_zero();
}

RGBFilm *RGBFilm::create(const Filter *filter, const ParameterDictionary &parameters,
                         GPUMemoryAllocator &allocator) {
    auto resolution_x = parameters.get_integer("xresolution");
    auto resolution_y = parameters.get_integer("yresolution");

    auto film_resolution = Point2i(resolution_x, resolution_y);

    FloatType iso = 100;
    FloatType white_balance_val = 0.0;
    FloatType exposure_time = 1.0;
    FloatType imaging_ratio = exposure_time * iso / 100.0;

    auto d_illum = Spectrum::create_cie_d(white_balance_val == 0.0 ? 6500.0 : white_balance_val,
                                          CIE_S0, CIE_S1, CIE_S2, CIE_S_lambda, allocator);

    auto sensor = allocator.allocate<PixelSensor>();

    sensor->init_cie_1931(parameters.global_spectra->cie_xyz,
                          parameters.global_spectra->rgb_color_space,
                          white_balance_val == 0 ? nullptr : d_illum, imaging_ratio);

    auto gpu_pixels = allocator.allocate<Pixel>(film_resolution.x * film_resolution.y);

    {
        uint threads = 1024;
        uint blocks = divide_and_ceil(uint(film_resolution.x * film_resolution.y), threads);

        init_pixels<<<blocks, threads>>>(gpu_pixels, film_resolution);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    auto rgb_film = allocator.allocate<RGBFilm>();
    rgb_film->init(gpu_pixels, filter, sensor, film_resolution,
                   parameters.global_spectra->rgb_color_space);

    return rgb_film;
}

void RGBFilm::init(Pixel *_pixels, const Filter *_filter, const PixelSensor *_sensor,
                   const Point2i &_resolution, const RGBColorSpace *rgb_color_space) {
    pixels = _pixels;
    filter = _filter;
    sensor = _sensor;
    resolution = _resolution;

    pixel_bound = Bounds2i(Point2i(0, 0), Point2i(_resolution.x, _resolution.y));

    output_rgb_from_sensor_rgb = rgb_color_space->rgb_from_xyz * sensor->xyz_from_sensor_rgb;

    filter_integral = _filter->get_integral();
}

PBRT_CPU_GPU
const Filter *RGBFilm::get_filter() const {
    return filter;
}

PBRT_CPU_GPU
Bounds2f RGBFilm::sample_bounds() const {
    Vector2f radius = filter->get_radius();

    return Bounds2f(pixel_bound.p_min.to_point2f() - radius + Vector2f(0.5f, 0.5f),
                    pixel_bound.p_max.to_point2f() + radius - Vector2f(0.5f, 0.5f));
}

PBRT_CPU_GPU
void RGBFilm::add_sample(uint pixel_index, const SampledSpectrum &radiance_l,
                         const SampledWavelengths &lambda, FloatType weight) {
    if (weight == 0) {
        return;
    }

    auto rgb = sensor->to_sensor_rgb(radiance_l, lambda);

    if constexpr (DEBUG_MODE && rgb.has_nan()) {
        printf("RGBFilm::%s(): pixel(%d): has a NAN component\n", __func__, pixel_index);
    }

    // TODO: should I clamp pixel rgb?

    pixels[pixel_index].rgb_sum += weight * rgb;
    pixels[pixel_index].weight_sum += weight;
}

void RGBFilm::add_splat(const Point2f &p_film, const SampledSpectrum &radiance_l,
                        const SampledWavelengths &lambda) {
    auto rgb = sensor->to_sensor_rgb(radiance_l, lambda);
    // Compute bounds of affected pixels for splat, _splatBounds_
    Point2f pDiscrete = p_film + Vector2f(0.5, 0.5);
    Vector2f radius = filter->get_radius();

    Bounds2i splatBounds((pDiscrete - radius).floor(),
                         (pDiscrete + radius).floor() + Vector2i(1, 1));

    splatBounds = splatBounds.intersect(pixel_bound);

    for (const auto pi : splatBounds.range()) {
        // Evaluate filter at _pi_ and add splat contribution
        auto weight = filter->evaluate(Point2f(p_film - pi.to_point2f() - Vector2f(0.5, 0.5)));
        if (weight == 0) {
            continue;
        }

        auto pixel_index = pi.x + pi.y * resolution.x;

        pixels[pixel_index].rgb_splat += weight * rgb;
    }
}

PBRT_CPU_GPU
RGB RGBFilm::get_pixel_rgb(const Point2i p, FloatType splat_scale) const {
    const int idx = p.x + p.y * resolution.x;

    const Pixel &pixel = pixels[idx];

    auto rgb = pixel.rgb_sum;
    if (pixel.weight_sum != 0) {
        rgb /= pixel.weight_sum;
    }

    // Add splat value at pixel
    rgb += pixel.rgb_splat * (splat_scale / filter_integral);

    return output_rgb_from_sensor_rgb * rgb;
}

#include <pbrt/base/filter.h>
#include <pbrt/film/pixel_sensor.h>
#include <pbrt/film/rgb_film.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/spectrum_util/rgb_color_space.h>

static __global__ void init_pixels(Pixel *pixels, Point2i dimension) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= dimension.x * dimension.y) {
        return;
    }

    pixels[idx].init_zero();
}

RGBFilm::RGBFilm(const Filter *_filter, const ParameterDictionary &parameters,
                 GPUMemoryAllocator &allocator)
    : filter(_filter), filter_integral(_filter->get_integral()), sensor(nullptr) {
    auto resolution_x = parameters.get_integer("xresolution");
    auto resolution_y = parameters.get_integer("yresolution");

    resolution = Point2i(resolution_x, resolution_y);
    pixel_bound = Bounds2i(Point2i(0, 0), Point2i(resolution.x, resolution.y));

    Real iso = 100;
    Real white_balance_val = 0.0;
    Real exposure_time = 1.0;
    Real imaging_ratio = exposure_time * iso / 100.0;

    auto d_illum = Spectrum::create_cie_d(white_balance_val == 0.0 ? 6500.0 : white_balance_val,
                                          CIE_S0, CIE_S1, CIE_S2, CIE_S_lambda, allocator);

    auto _sensor = allocator.allocate<PixelSensor>();
    _sensor->init_cie_1931(parameters.global_spectra->cie_xyz,
                           parameters.global_spectra->rgb_color_space,
                           white_balance_val == 0 ? nullptr : d_illum, imaging_ratio);
    sensor = _sensor;

    pixels = allocator.allocate<Pixel>(resolution.x * resolution.y);

    int blocks = divide_and_ceil(int(resolution.x * resolution.y), MAX_THREADS_PER_BLOCKS);
    init_pixels<<<blocks, MAX_THREADS_PER_BLOCKS>>>(pixels, resolution);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    output_rgb_from_sensor_rgb =
        parameters.global_spectra->rgb_color_space->rgb_from_xyz * sensor->xyz_from_sensor_rgb;
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
void RGBFilm::add_sample(int pixel_index, const SampledSpectrum &radiance_l,
                         const SampledWavelengths &lambda, Real weight) {
    if (weight == 0) {
        return;
    }

    auto rgb = sensor->to_sensor_rgb(radiance_l, lambda);

    if (DEBUG_MODE && rgb.has_nan()) {
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
RGB RGBFilm::get_pixel_rgb(const Point2i p, Real splat_scale) const {
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

#pragma once

#include <pbrt/euclidean_space/bounds2.h>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/euclidean_space/squared_matrix.h>
#include <pbrt/spectrum_util/rgb.h>

class Filter;
class GPUMemoryAllocator;
class ParameterDictionary;
class PixelSensor;
class RGBColorSpace;
class SampledSpectrum;
class SampledWavelengths;

struct Pixel {
    RGB rgb_sum = RGB(0, 0, 0);
    Real weight_sum = 0;
    RGB rgb_splat = RGB(0, 0, 0);

    PBRT_CPU_GPU
    void init_zero() {
        rgb_sum = RGB(0, 0, 0);
        weight_sum = 0;

        rgb_splat = RGB(0, 0, 0);
    }
};

class RGBFilm {
  public:
    RGBFilm(const Filter *_filter, const ParameterDictionary &parameters,
            GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Point2i get_resolution() const {
        return resolution;
    }

    PBRT_CPU_GPU
    const Filter *get_filter() const;

    PBRT_CPU_GPU
    Bounds2f sample_bounds() const;

    PBRT_CPU_GPU
    void add_sample(int pixel_index, const SampledSpectrum &radiance_l,
                    const SampledWavelengths &lambda, Real weight);

    PBRT_CPU_GPU
    void add_sample(const Point2i &p_film, const SampledSpectrum &radiance_l,
                    const SampledWavelengths &lambda, Real weight) {
        int pixel_index = p_film.y * resolution.x + p_film.x;

        add_sample(pixel_index, radiance_l, lambda, weight);
    }

    void add_splat(const Point2f &p_film, const SampledSpectrum &radiance_l,
                   const SampledWavelengths &lambda);

    PBRT_CPU_GPU
    RGB get_pixel_rgb(const Point2i p, Real splat_scale = 1) const;

  private:
    Pixel *pixels = nullptr;
    const PixelSensor *sensor = nullptr;
    Point2i resolution = {0, 0};
    Bounds2i pixel_bound;

    const Filter *filter = nullptr;

    Real filter_integral = NAN;

    SquareMatrix<3> output_rgb_from_sensor_rgb;
};

#pragma once

#include <pbrt/euclidean_space/point2.h>
#include <pbrt/euclidean_space/squared_matrix.h>
#include <pbrt/spectrum_util/rgb.h>

class Filter;
class GPUMemoryAllocator;
class ParameterDictionary;
class PixelSensor;
class RGBColorSpace;

struct Pixel {
    RGB rgb_sum;
    Real weight_sum;
    RGB rgb_splat;

    PBRT_CPU_GPU
    void init_zero() {
        rgb_sum = RGB(0, 0, 0);
        weight_sum = 0;

        rgb_splat = RGB(0, 0, 0);
    }
};

class RGBFilm {
  public:
    static RGBFilm *create(const Filter *filter, const ParameterDictionary &parameters,
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
    void add_sample(uint pixel_index, const SampledSpectrum &radiance_l,
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
    Pixel *pixels;
    const PixelSensor *sensor;
    Point2i resolution;
    Bounds2i pixel_bound;

    const Filter *filter;

    Real filter_integral;

    SquareMatrix<3> output_rgb_from_sensor_rgb;

    void init(Pixel *_pixels, const Filter *_filter, const PixelSensor *_sensor,
              const Point2i &_resolution, const RGBColorSpace *rgb_color_space);
};

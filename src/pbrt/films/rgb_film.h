#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/squared_matrix.h"
#include "pbrt/spectrum_util/rgb.h"

class Filter;
class ParameterDictionary;
class PixelSensor;
class RGBColorSpace;

struct Pixel {
    RGB rgb_sum;
    FloatType weight_sum;

    PBRT_CPU_GPU
    void init_zero() {
        rgb_sum = RGB(0.0, 0.0, 0.0);
        weight_sum = 0;
    }
};

class RGBFilm {
  public:
    static RGBFilm *create(const ParameterDictionary &parameters,
                           std::vector<void *> &gpu_dynamic_pointers);

    void init(Pixel *_pixels, const PixelSensor *_sensor, const Point2i &_resolution,
              const RGBColorSpace *rgb_color_space);

    PBRT_CPU_GPU
    Point2i get_resolution() const {
        return resolution;
    }

    PBRT_CPU_GPU
    void add_sample(uint pixel_index, const SampledSpectrum &radiance_l,
                    const SampledWavelengths &lambda, FloatType weight);

    PBRT_CPU_GPU
    void add_sample(const Point2i &p_film, const SampledSpectrum &radiance_l,
                    const SampledWavelengths &lambda, FloatType weight) {
        int pixel_index = p_film.y * resolution.x + p_film.x;

        add_sample(pixel_index, radiance_l, lambda, weight);
    }

    void add_splat(const Point2f &p_film, const SampledSpectrum &radiance_l,
                   const SampledWavelengths &lambda, FloatType weight, const Filter *filter);

    PBRT_CPU_GPU
    RGB get_pixel_rgb(const Point2i p) const;

  private:
    Pixel *pixels;
    // TODO: change Pixel* to Array2D<Pixel>
    const PixelSensor *sensor;
    Point2i resolution;
    Bounds2i pixel_bound;

    SquareMatrix<3> output_rgb_from_sensor_rgb;
};

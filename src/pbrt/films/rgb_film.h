#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/squared_matrix.cuh"
#include "pbrt/spectra/rgb.h"
#include "pbrt/spectra/rgb_color_space.h"
#include "pbrt/base/film.h"

struct Pixel {
    RGB rgb_sum;
    double weight_sum;

    PBRT_GPU Pixel() : rgb_sum(RGB(0.0, 0.0, 0.0)), weight_sum(0.0) {}
};

class RGBFilm : public Film {
  public:
    PBRT_GPU RGBFilm(Pixel *_pixels, const PixelSensor *_sensor, const Point2i &_resolution,
                     const RGBColorSpace *rgb_color_space)
        : pixels(_pixels), sensor(_sensor), resolution(_resolution) {
        output_rgb_from_sensor_rgb = rgb_color_space->rgb_from_xyz * sensor->xyz_from_sensor_rgb;
    }

    PBRT_GPU void add_sample(const Point2i &p_film, const SampledSpectrum &radiance_l,
                             const SampledWavelengths &lambda, double weight) override {
        int pixel_index = p_film.y * resolution.x + p_film.x;

        auto rgb = sensor->to_sensor_rgb(radiance_l, lambda);

        if (rgb.has_nan()) {
            printf("RGBFilm::add_sample(): pixel(%d, %d): has a NAN component\n", p_film.x,
                   p_film.y);
        }

        pixels[pixel_index].rgb_sum += weight * rgb;
        pixels[pixel_index].weight_sum += weight;
    }

    PBRT_GPU RGB get_pixel_rgb(const Point2i &p) const override {
        int idx = p.x + p.y * resolution.x;
        auto pixel_rgb = pixels[idx].rgb_sum;
        if (pixels[idx].weight_sum != 0) {
            pixel_rgb /= pixels[idx].weight_sum;
        }
        // TODO: do black dots come from non-clamped pixel?

        return output_rgb_from_sensor_rgb * pixel_rgb;
    }

  private:
    Pixel *pixels;
    const PixelSensor *sensor;
    Point2i resolution;
    SquareMatrix<3> output_rgb_from_sensor_rgb;
};

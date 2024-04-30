#pragma once

#include "pbrt/spectrum_util/sampled_wavelengths.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"
#include "pbrt/spectrum_util/rgb.h"

class RGBFilm;

class Film {
  public:
    enum class Type {
        rgb,
    };

    void init(RGBFilm *rgb_film);

    PBRT_CPU_GPU
    void add_sample(const Point2i &p_film, const SampledSpectrum &radiance_l,
                    const SampledWavelengths &lambda, FloatType weight);

    PBRT_CPU_GPU
    RGB get_pixel_rgb(const Point2i &p) const;

    void write_to_png(const std::string &filename, const Point2i &resolution) const;

  private:
    void *film_ptr;
    Type film_type;
};

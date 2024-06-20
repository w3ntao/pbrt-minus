#pragma once

#include <vector>

#include "pbrt/spectrum_util/sampled_wavelengths.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"
#include "pbrt/spectrum_util/rgb.h"

class ParameterDictionary;
class RGBFilm;

class Film {
  public:
    enum class Type {
        rgb,
    };

    static Film *create_rgb_film(const ParameterDictionary &parameters,
                                 std::vector<void *> &gpu_dynamic_pointers);

    void init(RGBFilm *rgb_film);

    PBRT_CPU_GPU
    void add_sample(const Point2i &p_film, const SampledSpectrum &radiance_l,
                    const SampledWavelengths &lambda, FloatType weight);

    PBRT_CPU_GPU
    RGB get_pixel_rgb(const Point2i &p) const;

    void write_to_png(const std::string &filename, const Point2i &resolution) const;

  private:
    Type type;
    void *ptr;
};

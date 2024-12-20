#pragma once

#include "pbrt/spectrum_util/rgb.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"
#include "pbrt/spectrum_util/sampled_wavelengths.h"
#include <vector>

class Filter;
class ParameterDictionary;
class RGBFilm;

class Film {
  public:
    enum class Type {
        rgb,
    };

    static Film *create_rgb_film(const Filter *filter, const ParameterDictionary &parameters,
                                 std::vector<void *> &gpu_dynamic_pointers);

    void init(RGBFilm *rgb_film);

    PBRT_CPU_GPU
    Point2i get_resolution() const;

    PBRT_CPU_GPU
    Bounds2f sample_bounds() const;

    PBRT_CPU_GPU
    void add_sample(uint pixel_index, const SampledSpectrum &radiance_l,
                    const SampledWavelengths &lambda, FloatType weight);

    PBRT_CPU_GPU
    void add_sample(const Point2i &p_film, const SampledSpectrum &radiance_l,
                    const SampledWavelengths &lambda, FloatType weight);

    // CPU only
    void add_splat(const Point2f &p_film, const SampledSpectrum &radiance_l,
                   const SampledWavelengths &lambda, FloatType weight);

    PBRT_CPU_GPU
    RGB get_pixel_rgb(const Point2i &p) const;

    void copy_to_frame_buffer(std::vector<uint8_t> &cpu_frame_buffer) const;

    void write_to_png(const std::string &filename) const;

  private:
    Type type;
    void *ptr;
};

#pragma once

#include <pbrt/spectrum_util/rgb.h>
#include <pbrt/spectrum_util/sampled_spectrum.h>
#include <pbrt/spectrum_util/sampled_wavelengths.h>
#include <vector>

class Filter;
class GPUMemoryAllocator;
class ParameterDictionary;
class RGBFilm;

class Film {
  public:
    enum class Type {
        rgb,
    };

    static Film *create_rgb_film(const Filter *filter, const ParameterDictionary &parameters,
                                 GPUMemoryAllocator &allocator);

    void init(RGBFilm *rgb_film);

    PBRT_CPU_GPU
    Point2i get_resolution() const;

    PBRT_CPU_GPU
    const Filter *get_filter() const;

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
                   const SampledWavelengths &lambda);

    PBRT_CPU_GPU
    RGB get_pixel_rgb(const Point2i &p, FloatType splat_scale = 1) const;

    void copy_to_frame_buffer(uint8_t *gpu_frame_buffer, FloatType splat_scale = 1) const;

    void write_to_png(const std::string &filename, FloatType splat_scale = 1) const;

  private:
    Type type;
    void *ptr;
};

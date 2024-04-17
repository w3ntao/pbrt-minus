#pragma once

#include "pbrt/spectra/sampled_wavelengths.h"
#include "pbrt/spectra/sampled_spectrum.h"
#include "pbrt/spectra/rgb.h"

class RGBFilm;

class Film {
  public:
    void init(RGBFilm *rgb_film);

    PBRT_CPU_GPU
    void add_sample(const Point2i &p_film, const SampledSpectrum &radiance_l,
                    const SampledWavelengths &lambda, FloatType weight);

    PBRT_CPU_GPU
    RGB get_pixel_rgb(const Point2i &p) const;

    void write_to_png(const std::string &filename, const Point2i &resolution) const;

  private:
    enum class FilmType { rgb };

    void *film_ptr;
    FilmType film_type;

    PBRT_CPU_GPU
    void report_error() const {
        printf("\nFilm: this type is not implemented\n");
#if defined(__CUDA_ARCH__)
        asm("trap;");
#else
        throw std::runtime_error("Film: this type is not implemented\n");
#endif
    }
};

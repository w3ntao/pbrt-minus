#pragma once

#include "pbrt/spectra/sampled_wavelengths.h"
#include "pbrt/spectra/sampled_spectrum.h"

class Film {
  public:
    PBRT_GPU virtual void add_sample(const Point2i &p_film, const SampledSpectrum &radiance_l,
                                     const SampledWavelengths &lambda, double weight) = 0;

    PBRT_GPU virtual void write_to_rgb(RGB *output_rgb, int idx) const = 0;
};

#pragma once

#include "pbrt/spectra/sampled_wavelengths.h"

class Spectrum {
  public:
    virtual ~Spectrum() = default;

    PBRT_CPU_GPU
    virtual double operator()(double lambda) const = 0;

    PBRT_CPU_GPU
    virtual SampledSpectrum Sample(const SampledWavelengths &lambda) const = 0;
};

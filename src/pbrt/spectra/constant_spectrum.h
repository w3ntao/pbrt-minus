#pragma once

#include <pbrt/gpu/macro.h>

class ConstantSpectrum {
  public:
    explicit ConstantSpectrum(const Real _c) : c(_c) {}

    PBRT_CPU_GPU
    Real operator()(Real) const {
        return c;
    }

    // ConstantSpectrum Public Methods
    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &) const {
        return SampledSpectrum(c);
    }

  private:
    Real c = NAN;
};

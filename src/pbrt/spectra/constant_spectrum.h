#pragma once

#include <pbrt/gpu/macro.h>

class ConstantSpectrum {
  public:
    void init(FloatType _c) {
        c = _c;
    }

    PBRT_CPU_GPU
    FloatType operator()(FloatType) const {
        return c;
    }

    // ConstantSpectrum Public Methods
    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &) const {
        return SampledSpectrum(c);
    }

  private:
    FloatType c;
};

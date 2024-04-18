#pragma once

#include "pbrt/base/spectrum.h"
#include "pbrt/base/texture.h"

class SpectrumConstantTexture {
  public:
    void init(const Spectrum *_value) {
        value = _value;
    }

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx,
                             const SampledWavelengths &lambda) const {
        return value->sample(lambda);
    }

  public:
    const Spectrum *value;
};

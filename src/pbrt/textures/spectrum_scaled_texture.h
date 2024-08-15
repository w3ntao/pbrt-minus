#pragma once

#include "pbrt/base/spectrum.h"
#include "pbrt/util/macro.h"

class SpectrumTexture;
class TextureEvalContext;
class SampledWavelengths;
class SampledSpectrum;
class ParameterDictionary;

class SpectrumScaledTexture {
  public:
    void init(SpectrumType spectrum_type, const ParameterDictionary &parameters,
              std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *texture;
    FloatType scale;
};

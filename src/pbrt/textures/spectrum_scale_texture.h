#pragma once

#include "pbrt/util/macro.h"

class SpectrumTexture;
class TextureEvalContext;
class SampledWavelengths;
class SampledSpectrum;

class SpectrumScaleTexture {
  public:
    void init(const SpectrumTexture *_texture, FloatType _scale);

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *texture;
    FloatType scale;
};

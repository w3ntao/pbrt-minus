#pragma once

#include <pbrt/base/spectrum.h>

class FloatTexture;
class SpectrumTexture;
struct TextureEvalContext;

class SpectrumMixTexture {
  public:
    SpectrumMixTexture(const ParameterDictionary &parameters, SpectrumType spectrum_type,
                       GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *tex1;
    const SpectrumTexture *tex2;
    const FloatTexture *amount;
};

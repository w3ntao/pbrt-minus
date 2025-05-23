#pragma once

#include <pbrt/gpu/macro.h>

class GPUMemoryAllocator;
class SpectrumTexture;
class SampledWavelengths;
class SampledSpectrum;
class ParameterDictionary;
struct TextureEvalContext;

class SpectrumScaledTexture {
  public:
    SpectrumScaledTexture(SpectrumType spectrum_type, const ParameterDictionary &parameters,
                          GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *texture = nullptr;
    Real scale = NAN;
};

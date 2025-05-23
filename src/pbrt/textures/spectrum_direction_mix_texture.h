#pragma once

#include <pbrt/gpu/macro.h>

class SampledSpectrum;
class SpectrumTexture;
struct TextureEvalContext;

class SpectrumDirectionMixTexture {
  public:
    SpectrumDirectionMixTexture(const Transform &render_from_texture,
                                const ParameterDictionary &parameters, SpectrumType spectrumType,
                                GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *tex1 = nullptr;
    const SpectrumTexture *tex2 = nullptr;
    Vector3f dir = Vector3f(NAN, NAN, NAN);
};

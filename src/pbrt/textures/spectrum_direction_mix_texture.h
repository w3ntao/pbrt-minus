#pragma once

#include <pbrt/gpu/macro.h>
#include <pbrt/spectrum_util/sampled_spectrum.h>

class SpectrumTexture;
struct TextureEvalContext;

class SpectrumDirectionMixTexture {
  public:
    static const SpectrumDirectionMixTexture *create(const Transform &render_from_texture,
                                                     const ParameterDictionary &parameters,
                                                     SpectrumType spectrumType,
                                                     GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *tex1;
    const SpectrumTexture *tex2;
    Vector3f dir;
};

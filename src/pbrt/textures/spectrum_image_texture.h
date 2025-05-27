#pragma once

#include <pbrt/base/spectrum.h>
#include <pbrt/textures/image_texture_base.h>

struct TextureEvalContext;

class SpectrumImageTexture : ImageTextureBase {
  public:
    SpectrumImageTexture(SpectrumType _spectrum_type, const Transform &render_from_object,
                         const RGBColorSpace *_color_space, const ParameterDictionary &parameters,
                         GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    SpectrumType spectrum_type;
    const RGBColorSpace *color_space = nullptr;
};

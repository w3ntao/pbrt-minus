#pragma once

#include "pbrt/base/spectrum.h"
#include "pbrt/euclidean_space/transform.h"
#include "pbrt/textures/image_texture_base.h"
#include "pbrt/textures/mipmap.h"

class SpectrumImageTexture : ImageTextureBase {
  public:
    static const SpectrumImageTexture *create(SpectrumType spectrum_type,
                                              const Transform &render_from_object,
                                              const RGBColorSpace *_color_space,
                                              const ParameterDictionary &parameters,
                                              std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    SpectrumType spectrum_type;
    const RGBColorSpace *color_space;

    void init(SpectrumType _spectrum_type, const Transform &render_from_object,
              const ParameterDictionary &parameters, std::vector<void *> &gpu_dynamic_pointers,
              const RGBColorSpace *_color_space);
};

#pragma once

#include "pbrt/base/spectrum.h"
#include "pbrt/euclidean_space/transform.h"

#include "pbrt/textures/image_texture_base.h"
#include "pbrt/textures/mipmap.h"

class SpectrumImageTexture : ImageTextureBase {
  public:
    void init(const ParameterDict &parameters, const RGBColorSpace *_color_space,
              std::vector<void *> &gpu_dynamic_pointers);
    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    SpectrumType spectrum_type;
    const RGBColorSpace *color_space;
};

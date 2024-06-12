#include "pbrt/textures/spectrum_scale_texture.h"

#include "pbrt/base/texture.h"
#include "pbrt/scene/parameter_dictionary.h"

void SpectrumScaleTexture::init(const ParameterDictionary &parameters) {
    texture = parameters.get_spectrum_texture("tex");
    scale = parameters.get_float("scale", 1.0);
}

PBRT_CPU_GPU
SampledSpectrum SpectrumScaleTexture::evaluate(const TextureEvalContext &ctx,
                                               const SampledWavelengths &lambda) const {
    if (scale == 0.0) {
        return 0.0;
    }

    return texture->evaluate(ctx, lambda) * scale;
}

#include "pbrt/textures/spectrum_scale_texture.h"
#include "pbrt/base/texture.h"

void SpectrumScaleTexture::init(const SpectrumTexture *_texture, FloatType _scale) {
    texture = _texture;
    scale = _scale;
}

PBRT_CPU_GPU
SampledSpectrum SpectrumScaleTexture::evaluate(const TextureEvalContext &ctx,
                                               const SampledWavelengths &lambda) const {
    if (scale == 0.0) {
        return 0.0;
    }
    
    return texture->evaluate(ctx, lambda) * scale;
}

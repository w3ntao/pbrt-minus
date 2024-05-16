#include "pbrt/base/texture.h"
#include "pbrt/textures/float_constant_texture.h"
#include "pbrt/textures/spectrum_constant_texture.h"

void FloatTexture::init(const FloatConstantTexture *float_constant_texture) {
    type = Type::constant;
    ptr = float_constant_texture;
}

void SpectrumTexture::init(const SpectrumConstantTexture *spectrum_constant_texture) {
    type = Type::constant;
    ptr = spectrum_constant_texture;
}

PBRT_CPU_GPU
FloatType FloatTexture::evaluate(const TextureEvalContext &ctx) const {
    switch (type) {
    case (Type::constant): {
        return ((FloatConstantTexture *)ptr)->evaluate(ctx);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
SampledSpectrum SpectrumTexture::evaluate(const TextureEvalContext &ctx,
                                          const SampledWavelengths &lambda) const {
    switch (type) {
    case (Type::constant): {
        return ((SpectrumConstantTexture *)ptr)->evaluate(ctx, lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

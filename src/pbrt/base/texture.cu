#include "pbrt/base/texture.h"
#include "pbrt/textures/float_constant_texture.h"
#include "pbrt/textures/spectrum_constant_texture.h"
#include "pbrt/textures/spectrum_image_texture.h"
#include "pbrt/textures/spectrum_scale_texture.h"

void FloatTexture::init(const FloatConstantTexture *constant_texture) {
    type = Type::constant;
    ptr = constant_texture;
}

void SpectrumTexture::init(const SpectrumConstantTexture *constant_texture) {
    type = Type::constant;
    ptr = constant_texture;
}

void SpectrumTexture::init(const SpectrumImageTexture *image_texture) {
    type = Type::image;
    ptr = image_texture;
}

void SpectrumTexture::init(const SpectrumScaleTexture *scale_texture) {
    type = Type::scale;
    ptr = scale_texture;
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
    case (Type::image): {
        return ((SpectrumImageTexture *)ptr)->evaluate(ctx, lambda);
    }
    case (Type::scale): {
        return ((SpectrumScaleTexture *)ptr)->evaluate(ctx, lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

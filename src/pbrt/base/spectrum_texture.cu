#include <pbrt/base/spectrum_texture.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/spectrum_util/rgb_color_space.h>
#include <pbrt/textures/spectrum_checkerboard_texture.h>
#include <pbrt/textures/spectrum_constant_texture.h>
#include <pbrt/textures/spectrum_image_texture.h>
#include <pbrt/textures/spectrum_mix_texture.h>
#include <pbrt/textures/spectrum_scaled_texture.h>

const SpectrumTexture *
SpectrumTexture::create(const std::string &texture_type, const SpectrumType spectrum_type,
                        const Transform &render_from_object, const RGBColorSpace *color_space,
                        const ParameterDictionary &parameters, GPUMemoryAllocator &allocator) {
    auto spectrum_texture = allocator.allocate<SpectrumTexture>();

    if (texture_type == "checkerboard") {
        auto checkerboard_texture = SpectrumCheckerboardTexture::create(
            render_from_object, spectrum_type, parameters, allocator);

        spectrum_texture->init(checkerboard_texture);

        return spectrum_texture;
    }

    if (texture_type == "constant") {
        auto constant_texture =
            SpectrumConstantTexture::create(parameters, spectrum_type, allocator);

        spectrum_texture->init(constant_texture);

        return spectrum_texture;
    }

    if (texture_type == "imagemap") {
        auto image_texture = SpectrumImageTexture::create(spectrum_type, render_from_object,
                                                          color_space, parameters, allocator);
        spectrum_texture->init(image_texture);
        return spectrum_texture;
    }

    if (texture_type == "mix") {
        auto mix_texture = SpectrumMixTexture::create(parameters, spectrum_type, allocator);

        spectrum_texture->init(mix_texture);
        return spectrum_texture;
    }

    if (texture_type == "scale") {
        auto scaled_texture = allocator.allocate<SpectrumScaledTexture>();

        scaled_texture->init(spectrum_type, parameters, allocator);
        spectrum_texture->init(scaled_texture);
        return spectrum_texture;
    }

    printf("\ntexture type `%s` not implemented for SpectrumTexture\n", texture_type.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

const SpectrumTexture *
SpectrumTexture::create_constant_float_val_texture(FloatType val, GPUMemoryAllocator &allocator) {
    auto spectrum_constant_texture = allocator.allocate<SpectrumConstantTexture>();
    auto spectrum_texture = allocator.allocate<SpectrumTexture>();

    spectrum_constant_texture->init(Spectrum::create_constant_spectrum(val, allocator));
    spectrum_texture->init(spectrum_constant_texture);

    return spectrum_texture;
}

const SpectrumTexture *SpectrumTexture::create_constant_texture(const Spectrum *spectrum,
                                                                GPUMemoryAllocator &allocator) {
    if (spectrum == nullptr) {
        REPORT_FATAL_ERROR();
    }
    auto spectrum_constant_texture = allocator.allocate<SpectrumConstantTexture>();
    auto spectrum_texture = allocator.allocate<SpectrumTexture>();

    spectrum_constant_texture->init(spectrum);
    spectrum_texture->init(spectrum_constant_texture);

    return spectrum_texture;
}

void SpectrumTexture::init(const SpectrumCheckerboardTexture *checkerboard_texture) {
    type = Type::checkerboard;
    ptr = checkerboard_texture;
}

void SpectrumTexture::init(const SpectrumConstantTexture *constant_texture) {
    type = Type::constant;
    ptr = constant_texture;
}

void SpectrumTexture::init(const SpectrumImageTexture *image_texture) {
    type = Type::image;
    ptr = image_texture;
}

void SpectrumTexture::init(const SpectrumMixTexture *mix_texture) {
    type = Type::mix;
    ptr = mix_texture;
}

void SpectrumTexture::init(const SpectrumScaledTexture *scale_texture) {
    type = Type::scaled;
    ptr = scale_texture;
}

PBRT_CPU_GPU
SampledSpectrum SpectrumTexture::evaluate(const TextureEvalContext &ctx,
                                          const SampledWavelengths &lambda) const {
    switch (type) {
    case Type::checkerboard: {
        return static_cast<const SpectrumCheckerboardTexture *>(ptr)->evaluate(ctx, lambda);
    }

    case Type::constant: {
        return static_cast<const SpectrumConstantTexture *>(ptr)->evaluate(ctx, lambda);
    }

    case Type::image: {
        return static_cast<const SpectrumImageTexture *>(ptr)->evaluate(ctx, lambda);
    }

    case Type::mix: {
        return static_cast<const SpectrumMixTexture *>(ptr)->evaluate(ctx, lambda);
    }

    case Type::scaled: {
        return static_cast<const SpectrumScaledTexture *>(ptr)->evaluate(ctx, lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

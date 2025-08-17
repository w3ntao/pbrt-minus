#include <pbrt/base/spectrum_texture.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/spectrum_util/rgb_color_space.h>
#include <pbrt/textures/spectrum_checkerboard_texture.h>
#include <pbrt/textures/spectrum_constant_texture.h>
#include <pbrt/textures/spectrum_direction_mix_texture.h>
#include <pbrt/textures/spectrum_image_texture.h>
#include <pbrt/textures/spectrum_mix_texture.h>
#include <pbrt/textures/spectrum_scaled_texture.h>

const SpectrumTexture *
SpectrumTexture::create(const std::string &texture_type, const SpectrumType spectrum_type,
                        const Transform &render_from_texture, const RGBColorSpace *color_space,
                        const ParameterDictionary &parameters, GPUMemoryAllocator &allocator) {
    if (texture_type == "checkerboard") {
        auto checkerboard_texture = allocator.create<SpectrumCheckerboardTexture>(
            render_from_texture, spectrum_type, parameters, allocator);

        return allocator.create<SpectrumTexture>(checkerboard_texture);
    }

    if (texture_type == "constant") {
        auto constant_texture =
            allocator.create<SpectrumConstantTexture>(parameters, spectrum_type, allocator);

        return allocator.create<SpectrumTexture>(constant_texture);
    }

    if (texture_type == "directionmix") {
        auto direction_mix_texture = allocator.create<SpectrumDirectionMixTexture>(
            render_from_texture, parameters, spectrum_type, allocator);

        return allocator.create<SpectrumTexture>(direction_mix_texture);
    }

    if (texture_type == "imagemap") {
        auto image_texture = allocator.create<SpectrumImageTexture>(
            spectrum_type, render_from_texture, color_space, parameters, allocator);

        return allocator.create<SpectrumTexture>(image_texture);
    }

    if (texture_type == "mix") {
        auto mix_texture =
            allocator.create<SpectrumMixTexture>(parameters, spectrum_type, allocator);

        return allocator.create<SpectrumTexture>(mix_texture);
    }

    if (texture_type == "scale") {
        auto scaled_texture =
            allocator.create<SpectrumScaledTexture>(spectrum_type, parameters, allocator);

        return allocator.create<SpectrumTexture>(scaled_texture);
    }

    printf("\ntexture type `%s` not implemented for SpectrumTexture\n", texture_type.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

const SpectrumTexture *
SpectrumTexture::create_constant_float_val_texture(Real val, GPUMemoryAllocator &allocator) {
    const auto spectrum = Spectrum::create_constant_spectrum(val, allocator);
    return create_constant_texture(spectrum, allocator);
}

const SpectrumTexture *SpectrumTexture::create_constant_texture(const Spectrum *spectrum,
                                                                GPUMemoryAllocator &allocator) {
    if (spectrum == nullptr) {
        REPORT_FATAL_ERROR();
    }

    auto constant_texture = allocator.create<SpectrumConstantTexture>(spectrum);

    return allocator.create<SpectrumTexture>(constant_texture);
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

    case Type::direction_mix: {
        return static_cast<const SpectrumDirectionMixTexture *>(ptr)->evaluate(ctx, lambda);
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

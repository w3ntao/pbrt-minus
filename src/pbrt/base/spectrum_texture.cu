#include <pbrt/base/spectrum_texture.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/spectrum_util/rgb_color_space.h>

const SpectrumTexture *
SpectrumTexture::create(const std::string &texture_type, const SpectrumType spectrum_type,
                        const Transform &render_from_texture, const RGBColorSpace *color_space,
                        const ParameterDictionary &parameters, GPUMemoryAllocator &allocator) {
    auto spectrum_texture = allocator.allocate<SpectrumTexture>();

    if (texture_type == "checkerboard") {
        *spectrum_texture =
            SpectrumCheckerboardTexture(render_from_texture, spectrum_type, parameters, allocator);
        return spectrum_texture;
    }

    if (texture_type == "constant") {
        *spectrum_texture = SpectrumConstantTexture(parameters, spectrum_type, allocator);
        return spectrum_texture;
    }

    if (texture_type == "directionmix") {
        *spectrum_texture =
            SpectrumDirectionMixTexture(render_from_texture, parameters, spectrum_type, allocator);
        return spectrum_texture;
    }

    if (texture_type == "imagemap") {
        *spectrum_texture = SpectrumImageTexture(spectrum_type, render_from_texture, color_space,
                                                 parameters, allocator);
        return spectrum_texture;
    }

    if (texture_type == "mix") {
        *spectrum_texture = SpectrumMixTexture(parameters, spectrum_type, allocator);
        return spectrum_texture;
    }

    if (texture_type == "scale") {
        *spectrum_texture = SpectrumScaledTexture(spectrum_type, parameters, allocator);
        return spectrum_texture;
    }

    printf("\ntexture type `%s` not implemented for SpectrumTexture\n", texture_type.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

const SpectrumTexture *
SpectrumTexture::create_constant_float_val_texture(Real val, GPUMemoryAllocator &allocator) {
    auto spectrum_texture = allocator.allocate<SpectrumTexture>();
    *spectrum_texture = SpectrumConstantTexture(Spectrum::create_constant_spectrum(val, allocator));
    return spectrum_texture;
}

const SpectrumTexture *SpectrumTexture::create_constant_texture(const Spectrum *spectrum,
                                                                GPUMemoryAllocator &allocator) {
    if (spectrum == nullptr) {
        REPORT_FATAL_ERROR();
    }

    auto spectrum_texture = allocator.allocate<SpectrumTexture>();
    *spectrum_texture = SpectrumConstantTexture(spectrum);

    return spectrum_texture;
}

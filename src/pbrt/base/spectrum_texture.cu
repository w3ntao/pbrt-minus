#include "pbrt/base/spectrum_texture.h"
#include "pbrt/spectra/constant_spectrum.h"
#include "pbrt/spectrum_util/rgb_color_space.h"
#include "pbrt/textures/spectrum_constant_texture.h"
#include "pbrt/textures/spectrum_image_texture.h"
#include "pbrt/textures/spectrum_scaled_texture.h"

const SpectrumTexture *SpectrumTexture::create(const std::string &texture_type,
                                               SpectrumType spectrum_type,
                                               const Transform &render_from_object,
                                               const RGBColorSpace *color_space,
                                               const ParameterDictionary &parameters,
                                               std::vector<void *> &gpu_dynamic_pointers) {
    if (texture_type == "imagemap") {
        auto image_texture = SpectrumImageTexture::create(
            spectrum_type, render_from_object, color_space, parameters, gpu_dynamic_pointers);

        SpectrumTexture *spectrum_texture;
        CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_texture, sizeof(SpectrumTexture)));
        gpu_dynamic_pointers.push_back(spectrum_texture);

        spectrum_texture->init(image_texture);
        return spectrum_texture;
    }

    if (texture_type == "scale") {
        SpectrumScaledTexture *scaled_texture;
        SpectrumTexture *spectrum_texture;

        CHECK_CUDA_ERROR(cudaMallocManaged(&scaled_texture, sizeof(SpectrumScaledTexture)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_texture, sizeof(SpectrumTexture)));

        gpu_dynamic_pointers.push_back(scaled_texture);
        gpu_dynamic_pointers.push_back(spectrum_texture);

        scaled_texture->init(spectrum_type, parameters, gpu_dynamic_pointers);
        spectrum_texture->init(scaled_texture);
        return spectrum_texture;
    }

    printf("\ntexture type `%s` not implemented for SpectrumTexture\n", texture_type.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

const SpectrumTexture *
SpectrumTexture::create_constant_float_val_texture(FloatType val,
                                                   std::vector<void *> &gpu_dynamic_pointers) {
    SpectrumConstantTexture *spectrum_constant_texture;
    SpectrumTexture *spectrum_texture;

    CHECK_CUDA_ERROR(
        cudaMallocManaged(&spectrum_constant_texture, sizeof(SpectrumConstantTexture)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_texture, sizeof(SpectrumTexture)));

    spectrum_constant_texture->init(Spectrum::create_constant_spectrum(val, gpu_dynamic_pointers));
    spectrum_texture->init(spectrum_constant_texture);

    gpu_dynamic_pointers.push_back(spectrum_constant_texture);
    gpu_dynamic_pointers.push_back(spectrum_texture);

    return spectrum_texture;
}

const SpectrumTexture *
SpectrumTexture::create_constant_texture(const Spectrum *spectrum,
                                         std::vector<void *> &gpu_dynamic_pointers) {
    if (spectrum == nullptr) {
        REPORT_FATAL_ERROR();
    }

    SpectrumConstantTexture *spectrum_constant_texture;
    SpectrumTexture *spectrum_texture;

    CHECK_CUDA_ERROR(
        cudaMallocManaged(&spectrum_constant_texture, sizeof(SpectrumConstantTexture)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_texture, sizeof(SpectrumTexture)));

    gpu_dynamic_pointers.push_back(spectrum_constant_texture);
    gpu_dynamic_pointers.push_back(spectrum_texture);

    spectrum_constant_texture->init(spectrum);
    spectrum_texture->init(spectrum_constant_texture);

    return spectrum_texture;
}

void SpectrumTexture::init(const SpectrumConstantTexture *constant_texture) {
    type = Type::constant;
    ptr = constant_texture;
}

void SpectrumTexture::init(const SpectrumImageTexture *image_texture) {
    type = Type::image;
    ptr = image_texture;
}

void SpectrumTexture::init(const SpectrumScaledTexture *scale_texture) {
    type = Type::scaled;
    ptr = scale_texture;
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
    case (Type::scaled): {
        return ((SpectrumScaledTexture *)ptr)->evaluate(ctx, lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

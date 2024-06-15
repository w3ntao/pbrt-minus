#include "pbrt/base/texture.h"

#include "pbrt/spectra/constant_spectrum.h"
#include "pbrt/spectrum_util/rgb_color_space.h"

#include "pbrt/textures/float_constant_texture.h"
#include "pbrt/textures/spectrum_constant_texture.h"
#include "pbrt/textures/spectrum_image_texture.h"
#include "pbrt/textures/spectrum_scale_texture.h"

const FloatTexture *FloatTexture::create(FloatType val, std::vector<void *> &gpu_dynamic_pointers) {
    FloatConstantTexture *float_constant_texture;
    FloatTexture *float_texture;

    CHECK_CUDA_ERROR(cudaMallocManaged(&float_constant_texture, sizeof(FloatConstantTexture)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&float_texture, sizeof(FloatTexture)));

    float_constant_texture->init(val);
    float_texture->init(float_constant_texture);

    gpu_dynamic_pointers.push_back(float_constant_texture);
    gpu_dynamic_pointers.push_back(float_texture);

    return float_texture;
}

void FloatTexture::init(const FloatConstantTexture *constant_texture) {
    type = Type::constant;
    ptr = constant_texture;
}

const SpectrumTexture *SpectrumTexture::create(const std::string &type_of_texture,
                                               const ParameterDictionary &parameters,
                                               const RGBColorSpace *color_space,
                                               std::vector<void *> &gpu_dynamic_pointers) {
    if (type_of_texture == "imagemap") {
        auto image_texture =
            SpectrumImageTexture::create(parameters, color_space, gpu_dynamic_pointers);

        SpectrumTexture *spectrum_texture;
        CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_texture, sizeof(SpectrumTexture)));
        gpu_dynamic_pointers.push_back(spectrum_texture);

        spectrum_texture->init(image_texture);
        return spectrum_texture;
    }

    if (type_of_texture == "scale") {
        SpectrumScaleTexture *scale_texture;
        SpectrumTexture *spectrum_texture;

        CHECK_CUDA_ERROR(cudaMallocManaged(&scale_texture, sizeof(SpectrumScaleTexture)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_texture, sizeof(SpectrumTexture)));

        gpu_dynamic_pointers.push_back(scale_texture);
        gpu_dynamic_pointers.push_back(spectrum_texture);

        scale_texture->init(parameters);
        spectrum_texture->init(scale_texture);
        return spectrum_texture;
    }

    printf("\nTexture `%s` not implemented\n", type_of_texture.c_str());

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

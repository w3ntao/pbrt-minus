#include "pbrt/base/float_texture.h"
#include "pbrt/base/texture_eval_context.h"
#include "pbrt/textures/float_constant_texture.h"
#include "pbrt/textures/float_image_texture.h"
#include "pbrt/textures/float_scaled_texture.h"

void FloatTexture::init(const FloatConstantTexture *float_constant_texture) {
    type = Type::constant;
    ptr = float_constant_texture;
}

void FloatTexture::init(const FloatImageTexture *float_image_texture) {
    type = Type::image;
    ptr = float_image_texture;
}

void FloatTexture::init(const FloatScaledTexture *float_scaled_texture) {
    type = Type::scale;
    ptr = float_scaled_texture;
}

const FloatTexture *FloatTexture::create(const std::string &texture_type,
                                         const Transform &render_from_object,
                                         const ParameterDictionary &parameters,
                                         std::vector<void *> &gpu_dynamic_pointers) {
    if (texture_type == "constant") {
        auto constant_texture = FloatConstantTexture::create(parameters, gpu_dynamic_pointers);

        FloatTexture *float_texture;
        CHECK_CUDA_ERROR(cudaMallocManaged(&float_texture, sizeof(FloatTexture)));
        gpu_dynamic_pointers.push_back(float_texture);

        float_texture->init(constant_texture);

        return float_texture;
    }

    if (texture_type == "scale") {
        auto float_scaled_texture = FloatScaledTexture::create(parameters, gpu_dynamic_pointers);

        FloatTexture *float_texture;
        CHECK_CUDA_ERROR(cudaMallocManaged(&float_texture, sizeof(FloatTexture)));
        gpu_dynamic_pointers.push_back(float_texture);

        float_texture->init(float_scaled_texture);

        return float_texture;
    }

    if (texture_type == "imagemap") {
        auto float_image_texture =
            FloatImageTexture::create(render_from_object, parameters, gpu_dynamic_pointers);

        FloatTexture *float_texture;
        CHECK_CUDA_ERROR(cudaMallocManaged(&float_texture, sizeof(FloatTexture)));
        gpu_dynamic_pointers.push_back(float_texture);

        float_texture->init(float_image_texture);

        return float_texture;
    }

    printf("\ntexture type `%s` not implemented for FloatTexture\n", texture_type.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

const FloatTexture *
FloatTexture::create_constant_float_texture(FloatType val,
                                            std::vector<void *> &gpu_dynamic_pointers) {
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

PBRT_CPU_GPU
FloatType FloatTexture::evaluate(const TextureEvalContext &ctx) const {
    switch (type) {
    case (Type::constant): {
        return ((FloatConstantTexture *)ptr)->evaluate(ctx);
    }

    case (Type::image): {
        return ((FloatImageTexture *)ptr)->evaluate(ctx);
    }

    case (Type::scale): {
        return ((FloatScaledTexture *)ptr)->evaluate(ctx);
    }
    }

    REPORT_FATAL_ERROR();
    return NAN;
}
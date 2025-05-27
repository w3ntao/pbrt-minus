#include <pbrt/base/float_texture.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/textures/float_constant_texture.h>
#include <pbrt/textures/float_image_texture.h>
#include <pbrt/textures/float_scaled_texture.h>

const FloatTexture *FloatTexture::create(const std::string &texture_type,
                                         const Transform &render_from_object,
                                         const ParameterDictionary &parameters,
                                         GPUMemoryAllocator &allocator) {
    auto float_texture = allocator.allocate<FloatTexture>();

    if (texture_type == "constant") {
        auto constant_texture = allocator.allocate<FloatConstantTexture>();
        *constant_texture = FloatConstantTexture(parameters);
        float_texture->init(constant_texture);

        return float_texture;
    }

    if (texture_type == "scale") {
        auto scaled_texture = allocator.allocate<FloatScaledTexture>();
        *scaled_texture = FloatScaledTexture(parameters, allocator);
        float_texture->init(scaled_texture);

        return float_texture;
    }

    if (texture_type == "imagemap") {
        auto image_texture = allocator.allocate<FloatImageTexture>();
        *image_texture = FloatImageTexture(render_from_object, parameters, allocator);
        float_texture->init(image_texture);

        return float_texture;
    }

    printf("\ntexture type `%s` not implemented for FloatTexture\n", texture_type.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

const FloatTexture *FloatTexture::create_constant_float_texture(Real val,
                                                                GPUMemoryAllocator &allocator) {
    auto constant_texture = allocator.allocate<FloatConstantTexture>();
    *constant_texture = FloatConstantTexture(val);

    auto float_texture = allocator.allocate<FloatTexture>();
    float_texture->init(constant_texture);

    return float_texture;
}

PBRT_CPU_GPU
Real FloatTexture::evaluate(const TextureEvalContext &ctx) const {
    switch (type) {
    case Type::constant: {
        return static_cast<const FloatConstantTexture *>(ptr)->evaluate(ctx);
    }

    case Type::image: {
        return static_cast<const FloatImageTexture *>(ptr)->evaluate(ctx);
    }

    case Type::scale: {
        return static_cast<const FloatScaledTexture *>(ptr)->evaluate(ctx);
    }
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

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

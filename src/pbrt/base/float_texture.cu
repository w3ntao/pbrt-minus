#include <pbrt/base/float_texture.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/textures/float_constant_texture.h>
#include <pbrt/textures/float_image_texture.h>
#include <pbrt/textures/float_scaled_texture.h>

const FloatTexture *FloatTexture::create(const std::string &texture_type,
                                         const Transform &render_from_object,
                                         const ParameterDictionary &parameters,
                                         GPUMemoryAllocator &allocator) {
    if (texture_type == "constant") {
        auto constant_texture = allocator.create<FloatConstantTexture>(parameters);

        return allocator.create<FloatTexture>(constant_texture);
    }

    if (texture_type == "imagemap") {
        auto image_texture =
            allocator.create<FloatImageTexture>(render_from_object, parameters, allocator);

        return allocator.create<FloatTexture>(image_texture);
    }

    if (texture_type == "scale") {
        auto scaled_texture = allocator.create<FloatScaledTexture>(parameters, allocator);

        return allocator.create<FloatTexture>(scaled_texture);
    }

    printf("\ntexture type `%s` not implemented for FloatTexture\n", texture_type.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

const FloatTexture *FloatTexture::create_constant_float_texture(Real val,
                                                                GPUMemoryAllocator &allocator) {
    auto constant_texture = allocator.create<FloatConstantTexture>(val);

    return allocator.create<FloatTexture>(constant_texture);
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

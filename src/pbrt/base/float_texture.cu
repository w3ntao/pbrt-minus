#include <pbrt/base/float_texture.h>

const FloatTexture *FloatTexture::create(const std::string &texture_type,
                                         const Transform &render_from_object,
                                         const ParameterDictionary &parameters,
                                         GPUMemoryAllocator &allocator) {
    auto float_texture = allocator.allocate<FloatTexture>();

    if (texture_type == "constant") {
        *float_texture = FloatConstantTexture(parameters);
        return float_texture;
    }

    if (texture_type == "scale") {
        *float_texture = FloatScaledTexture(parameters, allocator);
        return float_texture;
    }

    if (texture_type == "imagemap") {
        *float_texture = FloatImageTexture(render_from_object, parameters, allocator);
        return float_texture;
    }

    printf("\ntexture type `%s` not implemented for FloatTexture\n", texture_type.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

const FloatTexture *FloatTexture::create_constant_float_texture(Real val,
                                                                GPUMemoryAllocator &allocator) {
    auto float_texture = allocator.allocate<FloatTexture>();
    *float_texture = FloatConstantTexture(val);

    return float_texture;
}

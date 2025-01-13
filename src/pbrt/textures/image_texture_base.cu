#include <pbrt/euclidean_space/transform.h>
#include <pbrt/textures/image_texture_base.h>
#include <pbrt/textures/mipmap.h>

#include <pbrt/gpu/gpu_memory_allocator.h>

void ImageTextureBase::init_image_texture_base(const Transform &render_from_object,
                                               const ParameterDictionary &parameters,
                                               GPUMemoryAllocator &allocator) {
    mipmap = MIPMap::create(parameters, allocator);

    scale = parameters.get_float("scale", 1.0);
    invert = parameters.get_bool("invert", false);

    texture_mapping = nullptr;

    const std::string mapping = parameters.get_one_string("mapping", "uv");

    if (mapping == "uv") {
        auto _texture_mapping = allocator.allocate<TextureMapping2D>();
        auto uv_mapping = allocator.allocate<UVMapping>();

        uv_mapping->init(parameters);
        _texture_mapping->init(uv_mapping);

        texture_mapping = _texture_mapping;

        return;
    }

    printf("\ntexture mapping `%s` not implemented\n", mapping.c_str());

    REPORT_FATAL_ERROR();
}

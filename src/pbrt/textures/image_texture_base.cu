#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/textures/image_texture_base.h>
#include <pbrt/textures/mipmap.h>
#include <pbrt/textures/texture_mapping_2d.h>

ImageTextureBase::ImageTextureBase(const Transform &render_from_object,
                                   const ParameterDictionary &parameters,
                                   GPUMemoryAllocator &allocator) {
    mipmap = allocator.create<MIPMap>(parameters, allocator);

    scale = parameters.get_float("scale", 1.0);
    invert = parameters.get_bool("invert", false);

    texture_mapping = TextureMapping2D::create(render_from_object, parameters, allocator);
}

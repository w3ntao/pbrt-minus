#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/textures/image_texture_base.h>
#include <pbrt/textures/mipmap.h>

void ImageTextureBase::init_image_texture_base(const Transform &render_from_object,
                                               const ParameterDictionary &parameters,
                                               GPUMemoryAllocator &allocator) {
    mipmap = MIPMap::create(parameters, allocator);

    scale = parameters.get_float("scale", 1.0);
    invert = parameters.get_bool("invert", false);

    texture_mapping = TextureMapping2D::create(render_from_object, parameters, allocator);
}

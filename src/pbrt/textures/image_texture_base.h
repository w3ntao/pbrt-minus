#pragma once

#include <pbrt/textures/mipmap.h>
#include <pbrt/textures/texture_mapping_2d.h>

class GPUMemoryAllocator;
class MIPMap;
class Transform;

// ImageTextureBase Definition
class ImageTextureBase {
  protected:
    const TextureMapping2D *texture_mapping;
    Real scale;
    bool invert;
    const MIPMap *mipmap;

    void init_image_texture_base(const Transform &render_from_object,
                                 const ParameterDictionary &parameters,
                                 GPUMemoryAllocator &allocator);
};

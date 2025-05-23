#pragma once

#include <pbrt/textures/mipmap.h>
#include <pbrt/textures/texture_mapping_2d.h>

class GPUMemoryAllocator;
class MIPMap;
class Transform;

// ImageTextureBase Definition
class ImageTextureBase {
  protected:
    const TextureMapping2D *texture_mapping = nullptr;
    Real scale = NAN;
    bool invert = false;
    const MIPMap *mipmap = nullptr;

    ImageTextureBase(const Transform &render_from_object, const ParameterDictionary &parameters,
                     GPUMemoryAllocator &allocator);
};

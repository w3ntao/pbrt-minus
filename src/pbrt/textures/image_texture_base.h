#pragma once

#include <pbrt/gpu/macro.h>

class GPUMemoryAllocator;
class MIPMap;
class Transform;
class ParameterDictionary;
struct TextureMapping2D;

class ImageTextureBase {
  protected:
    const TextureMapping2D *texture_mapping = nullptr;
    const MIPMap *mipmap = nullptr;

    Real scale = NAN;
    bool invert = false;

    ImageTextureBase(const Transform &render_from_object, const ParameterDictionary &parameters,
                     GPUMemoryAllocator &allocator);
};

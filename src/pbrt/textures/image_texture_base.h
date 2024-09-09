#pragma once

#include "pbrt/textures/mipmap.h"
#include "pbrt/textures/texture_mapping_2d.h"
#include <string>
#include <vector>

class MIPMap;
class Transform;

// ImageTextureBase Definition
class ImageTextureBase {
  protected:
    const TextureMapping2D *texture_mapping;
    FloatType scale;
    bool invert;
    const MIPMap *mipmap;
    
    void init_image_texture_base(const Transform &render_from_object,
                                 const ParameterDictionary &parameters,
                                 std::vector<void *> &gpu_dynamic_pointers);
};

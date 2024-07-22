#pragma once

#include <string>
#include <vector>

#include "pbrt/textures/texture_mapping_2d.h"
#include "pbrt/textures/mipmap.h"

class MIPMap;

// ImageTextureBase Definition
class ImageTextureBase {
  protected:
    UVMapping mapping;
    // TODO: change UVMapping to TextureMapping2D
    FloatType scale;
    bool invert;
    const MIPMap *mipmap;

    void init_image_texture_base(const ParameterDictionary &parameters,
                                 std::vector<void *> &gpu_dynamic_pointers);
};

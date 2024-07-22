#include "pbrt/textures/image_texture_base.h"
#include "pbrt/textures/mipmap.h"

void ImageTextureBase::init_image_texture_base(const ParameterDictionary &parameters,
                                               std::vector<void *> &gpu_dynamic_pointers) {
    mipmap = MIPMap::create(parameters, gpu_dynamic_pointers);

    mapping = UVMapping(parameters);

    scale = parameters.get_float("scale", 1.0);
    invert = parameters.get_bool("invert", false);
}

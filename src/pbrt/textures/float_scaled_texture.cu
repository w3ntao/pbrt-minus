#include "pbrt/textures/float_scaled_texture.h"

#include "pbrt/scene/parameter_dictionary.h"

const FloatScaledTexture *FloatScaledTexture::create(const ParameterDictionary &parameters,
                                                     std::vector<void *> &gpu_dynamic_pointers) {

    FloatScaledTexture *float_scaled_texture;
    CHECK_CUDA_ERROR(cudaMallocManaged(&float_scaled_texture, sizeof(FloatScaledTexture)));
    gpu_dynamic_pointers.push_back(float_scaled_texture);

    float_scaled_texture->scale = parameters.get_float("scale", 1.0);
    float_scaled_texture->texture = parameters.get_float_texture("tex");

    return float_scaled_texture;
}

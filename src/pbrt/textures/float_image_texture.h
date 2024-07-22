#pragma once

#include "pbrt/textures/image_texture_base.h"

class FloatImageTexture : ImageTextureBase {
  public:
    static const FloatImageTexture *create(const ParameterDictionary &parameters,
                                           std::vector<void *> &gpu_dynamic_pointers) {
        FloatImageTexture *float_image_texture;
        CHECK_CUDA_ERROR(cudaMallocManaged(&float_image_texture, sizeof(FloatImageTexture)));
        gpu_dynamic_pointers.push_back(float_image_texture);

        float_image_texture->init_image_texture_base(parameters, gpu_dynamic_pointers);

        return float_image_texture;
    }
};

#pragma once

#include "pbrt/base/spectrum.h"
#include "pbrt/base/spectrum_texture.h"
#include "pbrt/scene/parameter_dictionary.h"

class FloatConstantTexture {
  public:
    static const FloatConstantTexture *create(const ParameterDictionary &parameters,
                                              std::vector<void *> &gpu_dynamic_pointers) {
        FloatConstantTexture *texture;
        CHECK_CUDA_ERROR(cudaMallocManaged(&texture, sizeof(FloatConstantTexture)));
        gpu_dynamic_pointers.push_back(texture);

        texture->init(parameters.get_float("value", 1.0));

        return texture;
    }

    void init(FloatType _value) {
        value = _value;
    }

    PBRT_CPU_GPU
    FloatType evaluate(const TextureEvalContext &ctx) const {
        return value;
    }

  public:
    FloatType value;
};

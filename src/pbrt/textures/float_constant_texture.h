#pragma once

#include <pbrt/base/spectrum_texture.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>

class FloatConstantTexture {
  public:
    static const FloatConstantTexture *create(const ParameterDictionary &parameters,
                                              GPUMemoryAllocator &allocator) {
        auto texture = allocator.allocate<FloatConstantTexture>();

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

  private:
    FloatType value;
};

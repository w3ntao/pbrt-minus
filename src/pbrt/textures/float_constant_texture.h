#pragma once

// TODO: move implementation to .cu file
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>

struct TextureEvalContext;

class FloatConstantTexture {
  public:
    FloatConstantTexture(const Real _value) : value(_value) {}

    FloatConstantTexture(const ParameterDictionary &parameters) {
        value = parameters.get_float("value", 1.0);
    }

    PBRT_CPU_GPU
    Real evaluate(const TextureEvalContext &ctx) const {
        return value;
    }

  private:
    Real value = NAN;
};

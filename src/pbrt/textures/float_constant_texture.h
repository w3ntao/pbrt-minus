#pragma once

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

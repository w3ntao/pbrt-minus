#pragma once

#include "pbrt/base/spectrum.h"
#include "pbrt/base/texture.h"

class FloatConstantTexture {
  public:
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

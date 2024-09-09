#pragma once

#include "pbrt/util/macro.h"
#include <vector>

class FloatTexture;
class ParameterDictionary;
class TextureEvalContext;

class FloatScaledTexture {
  public:
    static const FloatScaledTexture *create(const ParameterDictionary &parameters,
                                            std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU
    FloatType evaluate(const TextureEvalContext &ctx) const;

  private:
    const FloatTexture *texture;
    const FloatTexture *scale;
};

#pragma once

#include "pbrt/util/macro.h"
#include <vector>

class FloatTexture;
class ParameterDictionary;

class FloatScaledTexture {
  public:
    static const FloatScaledTexture *create(const ParameterDictionary &parameters,
                                            std::vector<void *> &gpu_dynamic_pointers);

  private:
    const FloatTexture *texture;
    FloatType scale;
};

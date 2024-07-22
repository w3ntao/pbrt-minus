#pragma once

#include <vector>

#include "pbrt/util/macro.h"

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

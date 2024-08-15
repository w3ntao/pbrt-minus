#pragma once

#include <vector>

class ParameterDictionary;

class MixMaterial {
  public:
    void init(const ParameterDictionary &parameters, std::vector<void *> &gpu_dynamic_pointers);
};

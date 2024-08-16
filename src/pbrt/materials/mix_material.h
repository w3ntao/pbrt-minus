#pragma once

#include <vector>
#include "pbrt/util/macro.h"
#include "pbrt/base/material.h"

class ParameterDictionary;

class MixMaterial {
  public:
    void init(const ParameterDictionary &parameters);

    PBRT_CPU_GPU
    const Material *get_material(const SurfaceInteraction *si) const;

  private:
    FloatType amount;
    const Material *materials[2];
};

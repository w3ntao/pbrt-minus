#pragma once

#include <pbrt/base/material.h>
#include <pbrt/gpu/macro.h>
#include <vector>

class ParameterDictionary;

class MixMaterial {
  public:
    void init(const ParameterDictionary &parameters);

    PBRT_CPU_GPU
    const Material *get_material(const SurfaceInteraction *si) const;

  private:
    Real amount;
    const Material *materials[2];
};

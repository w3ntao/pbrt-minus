#pragma once

#include <pbrt/gpu/macro.h>

class Material;
class ParameterDictionary;
class SurfaceInteraction;

class MixMaterial {
  public:
    MixMaterial(const ParameterDictionary &parameters);

    // TODO: rewrite get_material to take in a hashed float rather than SurfaceInteraction
    PBRT_CPU_GPU
    const Material *get_material(const SurfaceInteraction *si) const;

  private:
    Real amount = 0.5;
    const Material *materials[2];
};

#pragma once

#include <pbrt/gpu/macro.h>

class BxDF;
class Material;
class MaterialEvalContext;
class ParameterDictionary;

class MixMaterial {
  public:
    static const MixMaterial *create(const ParameterDictionary &parameters,
                                     GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    const Material *get_material(const Real u) const;

    PBRT_CPU_GPU
    BxDF get_bxdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

  private:
    Real amount = 0.5;
    const Material *materials[2] = {nullptr, nullptr};

    MixMaterial(const ParameterDictionary &parameters);
};

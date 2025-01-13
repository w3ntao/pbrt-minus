#pragma once

#include <pbrt/gpu/macro.h>

class ConductorBxDF;
class FloatTexture;
class GPUMemoryAllocator;
class MaterialEvalContext;
class SampledWavelengths;
class SpectrumTexture;
class ParameterDictionary;

class ConductorMaterial {
  public:
    void init(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    ConductorBxDF get_conductor_bsdf(const MaterialEvalContext &ctx,
                                     SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *eta;
    const SpectrumTexture *k;
    const SpectrumTexture *reflectance;

    const FloatTexture *u_roughness;
    const FloatTexture *v_roughness;

    bool remap_roughness;
};

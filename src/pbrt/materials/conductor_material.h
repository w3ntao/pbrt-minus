#pragma once

#include <pbrt/gpu/macro.h>

class ConductorBxDF;
class FloatTexture;
class GPUMemoryAllocator;
class SampledWavelengths;
class SpectrumTexture;
class ParameterDictionary;
struct MaterialEvalContext;

class ConductorMaterial {
  public:
    ConductorMaterial(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    ConductorBxDF get_conductor_bsdf(const MaterialEvalContext &ctx,
                                     SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *eta = nullptr;
    const SpectrumTexture *k = nullptr;
    const SpectrumTexture *reflectance = nullptr;

    const FloatTexture *uRoughness = nullptr;
    const FloatTexture *vRoughness = nullptr;

    bool remapRoughness = true;
};

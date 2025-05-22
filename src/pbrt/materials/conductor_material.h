#pragma once

#include <pbrt/gpu/macro.h>

class BxDF;
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
    BxDF get_bxdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *eta = nullptr;
    const SpectrumTexture *k = nullptr;
    const SpectrumTexture *reflectance = nullptr;

    const FloatTexture *uRoughness = nullptr;
    const FloatTexture *vRoughness = nullptr;

    bool remapRoughness = true;
};

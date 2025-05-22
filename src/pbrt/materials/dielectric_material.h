#pragma once

#include <pbrt/gpu/macro.h>

class BxDF;
class FloatTexture;
class GPUMemoryAllocator;
class MaterialEvalContext;
class ParameterDictionary;
class SampledWavelengths;
class Spectrum;
class SpectrumTexture;

class DielectricMaterial {
  public:
    DielectricMaterial(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    BxDF get_bxdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

  private:
    const FloatTexture *uRoughness = nullptr;
    const FloatTexture *vRoughness = nullptr;

    const Spectrum *eta = nullptr;

    bool remapRoughness = true;
};

#pragma once

#include <pbrt/base/bsdf.h>
#include <pbrt/gpu/macro.h>

class DielectricBxDF;
class FloatTexture;
class GPUMemoryAllocator;
class MaterialEvalContext;
class ParameterDictionary;
class SampledWavelengths;
class Spectrum;
class SpectrumTexture;

class DielectricMaterial {
  public:
    void init(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    DielectricBxDF get_dielectric_bsdf(const MaterialEvalContext &ctx,
                                       SampledWavelengths &lambda) const;

  private:
    const FloatTexture *uRoughness;
    const FloatTexture *vRoughness;

    const Spectrum *eta;

    bool remapRoughness;
};

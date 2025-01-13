#pragma once

#include <pbrt/gpu/macro.h>

class CoatedConductorBxDF;
class FloatTexture;
class GPUMemoryAllocator;
class MaterialEvalContext;
class ParameterDictionary;
class SampledWavelengths;
class Spectrum;
class SpectrumTexture;

class CoatedConductorMaterial {
  public:
    void init(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    CoatedConductorBxDF get_coated_conductor_bsdf(const MaterialEvalContext &ctx,
                                                  SampledWavelengths &lambda) const;

  private:
    const FloatTexture *interfaceURoughness, *interfaceVRoughness, *thickness;
    const Spectrum *interfaceEta;
    const FloatTexture *g;
    const SpectrumTexture *albedo;
    const FloatTexture *conductorURoughness, *conductorVRoughness;
    const SpectrumTexture *conductorEta, *k, *reflectance;
    bool remapRoughness;
    int maxDepth, nSamples;
};

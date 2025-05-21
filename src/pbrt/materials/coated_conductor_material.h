#pragma once

#include <pbrt/gpu/macro.h>

class CoatedConductorBxDF;
class FloatTexture;
class GPUMemoryAllocator;
class ParameterDictionary;
class SampledWavelengths;
class Spectrum;
class SpectrumTexture;
struct MaterialEvalContext;

class CoatedConductorMaterial {
  public:
    CoatedConductorMaterial(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    CoatedConductorBxDF get_coated_conductor_bsdf(const MaterialEvalContext &ctx,
                                                  SampledWavelengths &lambda) const;

  private:
    const FloatTexture *interfaceURoughness = nullptr;
    const FloatTexture *interfaceVRoughness = nullptr;
    const FloatTexture *thickness = nullptr;

    const Spectrum *interfaceEta = nullptr;
    const FloatTexture *g = nullptr;
    const SpectrumTexture *albedo = nullptr;

    const FloatTexture *conductorURoughness = nullptr;
    const FloatTexture *conductorVRoughness = nullptr;

    const SpectrumTexture *conductorEta = nullptr;
    const SpectrumTexture *k = nullptr;
    const SpectrumTexture *reflectance = nullptr;

    bool remapRoughness = true;
    int maxDepth = 10;
    int nSamples = 1;
};

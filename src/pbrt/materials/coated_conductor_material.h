#pragma once

#include "pbrt/util/macro.h"
#include <vector>

class CoatedConductorBxDF;
class MaterialEvalContext;

class FloatTexture;
class ParameterDictionary;
class SampledWavelengths;
class Spectrum;
class SpectrumTexture;

class CoatedConductorMaterial {
  public:
    void init(const ParameterDictionary &parameters, std::vector<void *> &gpu_dynamic_pointers);

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

#pragma once

#include "pbrt/base/bsdf.h"
#include "pbrt/util/macro.h"
#include <vector>

class DielectricBxDF;
class FloatTexture;
class MaterialEvalContext;
class ParameterDictionary;
class SampledWavelengths;
class Spectrum;
class SpectrumTexture;

class DielectricMaterial {
  public:
    void init(const ParameterDictionary &parameters, std::vector<void *> &gpu_dynamic_pointers);

    PBRT_GPU
    DielectricBxDF get_dielectric_bsdf(const MaterialEvalContext &ctx,
                                       SampledWavelengths &lambda) const;

  private:
    const FloatTexture *uRoughness;
    const FloatTexture *vRoughness;

    const Spectrum *eta;

    bool remapRoughness;
};

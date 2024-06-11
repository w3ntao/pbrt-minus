#pragma once

#include <vector>
#include "pbrt/util/macro.h"

class ConductorBxDF;

class MaterialEvalContext;
class SampledWavelengths;

class FloatTexture;
class SpectrumTexture;

class ParameterDict;

class ConductorMaterial {
  public:
    void init(const ParameterDict &parameters, std::vector<void *> &gpu_dynamic_pointers);

    PBRT_GPU
    ConductorBxDF get_conductor_bsdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda);

  private:
    const SpectrumTexture *eta;
    const SpectrumTexture *k;
    const SpectrumTexture *reflectance;

    const FloatTexture *u_roughness;
    const FloatTexture *v_roughness;

    bool remap_roughness;
};

#pragma once

#include <vector>
#include "pbrt/bxdfs/coated_diffuse_bxdf.h"

class FloatTexture;
class MaterialEvalContext;
class Spectrum;
class SpectrumTexture;
class ParameterDict;

class CoatedDiffuseMaterial {
  public:
    void init(const ParameterDict &parameters, std::vector<void *> &gpu_dynamic_pointers);

    PBRT_GPU
    CoatedDiffuseBxDF get_coated_diffuse_bsdf(const MaterialEvalContext &ctx,
                                              SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *reflectance;
    const SpectrumTexture *albedo;

    const FloatTexture *u_roughness;
    const FloatTexture *v_roughness;
    const FloatTexture *thickness;
    const FloatTexture *g;

    const Spectrum *eta;

    bool remapRoughness;
    uint maxDepth;
    uint nSamples;
};

#pragma once

#include <pbrt/bxdfs/coated_diffuse_bxdf.h>

class FloatTexture;
class GPUMemoryAllocator;
class MaterialEvalContext;
class Spectrum;
class SpectrumTexture;
class ParameterDictionary;

class CoatedDiffuseMaterial {
  public:
    void init(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
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

#pragma once

class CoatedDiffuseBxDF;
class FloatTexture;
class GPUMemoryAllocator;
class Spectrum;
class SpectrumTexture;
class ParameterDictionary;
struct MaterialEvalContext;

class CoatedDiffuseMaterial {
  public:
    CoatedDiffuseMaterial(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    CoatedDiffuseBxDF get_coated_diffuse_bsdf(const MaterialEvalContext &ctx,
                                              SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *reflectance = nullptr;
    const SpectrumTexture *albedo = nullptr;

    const FloatTexture *u_roughness = nullptr;
    const FloatTexture *v_roughness = nullptr;
    const FloatTexture *thickness = nullptr;
    const FloatTexture *g = nullptr;

    const Spectrum *eta = nullptr;

    bool remapRoughness = true;
    uint maxDepth = 10;
    uint nSamples = 1;
};

#pragma once

class BxDF;
class FloatTexture;
class GPUMemoryAllocator;
class Spectrum;
class SpectrumTexture;
class ParameterDictionary;
struct MaterialEvalContext;

class CoatedDiffuseMaterial {
  public:
    static const CoatedDiffuseMaterial *create(const ParameterDictionary &parameters,
                                               GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    BxDF get_bxdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *reflectance = nullptr;
    const SpectrumTexture *albedo = nullptr;

    const FloatTexture *u_roughness = nullptr;
    const FloatTexture *v_roughness = nullptr;
    const FloatTexture *thickness = nullptr;
    const FloatTexture *g = nullptr;

    const Spectrum *eta = nullptr;

    bool remapRoughness = true;
    int maxDepth = 10;
    int nSamples = 1;

    CoatedDiffuseMaterial(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);
};

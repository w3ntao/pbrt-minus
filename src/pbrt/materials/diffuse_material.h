#pragma once

class BxDF;
class GPUMemoryAllocator;
class ParameterDictionary;
class RGBColorSpace;
class SampledWavelengths;
class SpectrumTexture;
struct MaterialEvalContext;

class DiffuseMaterial {
  public:
    explicit DiffuseMaterial(const SpectrumTexture *_reflectance) : reflectance(_reflectance) {}

    DiffuseMaterial(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    BxDF get_bxdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *reflectance = nullptr;
};

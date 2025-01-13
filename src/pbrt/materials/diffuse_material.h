#pragma once

#include <pbrt/base/bsdf.h>

class DiffuseBxDF;
class GPUMemoryAllocator;
class MaterialEvalContext;
class ParameterDictionary;
class RGBColorSpace;
class SampledWavelengths;
class SpectrumTexture;

class DiffuseMaterial {
  public:
    static const DiffuseMaterial *create(const SpectrumTexture *_reflectance,
                                         GPUMemoryAllocator &allocator);

    void init(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    DiffuseBxDF get_diffuse_bsdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *reflectance;
};

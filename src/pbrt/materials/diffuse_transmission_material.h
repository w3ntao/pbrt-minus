#pragma once

class BxDF;
class GPUMemoryAllocator;
class ParameterDictionary;
class SampledWavelengths;
class SpectrumTexture;
struct MaterialEvalContext;

class DiffuseTransmissionMaterial {
  public:
    static const DiffuseTransmissionMaterial *create(const ParameterDictionary &parameters,
                                                     GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    BxDF get_bxdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *reflectance = nullptr;
    const SpectrumTexture *transmittance = nullptr;
    Real scale = 1.0;

    DiffuseTransmissionMaterial(const ParameterDictionary &parameters,
                                GPUMemoryAllocator &allocator);
};

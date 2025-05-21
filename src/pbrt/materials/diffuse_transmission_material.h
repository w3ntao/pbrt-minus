#pragma once

class DiffuseTransmissionBxDF;
class GPUMemoryAllocator;
class ParameterDictionary;
class SampledWavelengths;
class SpectrumTexture;
struct MaterialEvalContext;

class DiffuseTransmissionMaterial {
  public:
    DiffuseTransmissionMaterial(const ParameterDictionary &parameters,
                                GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    DiffuseTransmissionBxDF get_diffuse_transmission_bsdf(const MaterialEvalContext &ctx,
                                                          SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *reflectance = nullptr;
    const SpectrumTexture *transmittance = nullptr;
    Real scale = 1.0;
};

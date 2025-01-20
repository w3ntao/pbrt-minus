#pragma once

class DiffuseTransmissionBxDF;
class GPUMemoryAllocator;
class MaterialEvalContext;
class ParameterDictionary;
class SampledWavelengths;
class SpectrumTexture;

class DiffuseTransmissionMaterial {
  public:
    static const DiffuseTransmissionMaterial *create(const ParameterDictionary &parameters,
                                                     GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    DiffuseTransmissionBxDF get_diffuse_transmission_bsdf(const MaterialEvalContext &ctx,
                                                          SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *reflectance;
    const SpectrumTexture *transmittance;
    FloatType scale;
};

#pragma once

#include <pbrt/base/bsdf.h>
#include <pbrt/base/spectrum_texture.h>
#include <pbrt/spectrum_util/sampled_spectrum.h>
#include <pbrt/gpu/macro.h>

class CoatedConductorMaterial;
class CoatedDiffuseMaterial;
class ConductorMaterial;
class DielectricMaterial;
class DiffuseMaterial;
class GPUMemoryAllocator;
class MixMaterial;

class ParameterDictionary;

class Material {
  public:
    enum class Type {
        coated_conductor,
        coated_diffuse,
        conductor,
        diffuse,
        dielectric,
        mix,
    };

    static const Material *create(const std::string &type_of_material,
                                  const ParameterDictionary &parameters,
                                  GPUMemoryAllocator &allocator);

    static const Material *create_diffuse_material(const SpectrumTexture *texture,
                                                   GPUMemoryAllocator &allocator);

    void init(const CoatedConductorMaterial *coated_conductor_material);

    void init(const CoatedDiffuseMaterial *coated_diffuse_material);

    void init(const ConductorMaterial *conductor_material);

    void init(const DielectricMaterial *dielectric_material);

    void init(const DiffuseMaterial *diffuse_material);

    void init(const MixMaterial *mix_material);

    PBRT_CPU_GPU
    Type get_material_type() const {
        return type;
    }

    PBRT_CPU_GPU
    const Material *get_mix_material(const SurfaceInteraction *si) const;

    PBRT_CPU_GPU
    CoatedConductorBxDF get_coated_conductor_bsdf(const MaterialEvalContext &ctx,
                                                  SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    CoatedDiffuseBxDF get_coated_diffuse_bsdf(const MaterialEvalContext &ctx,
                                              SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    ConductorBxDF get_conductor_bsdf(const MaterialEvalContext &ctx,
                                     SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    DielectricBxDF get_dielectric_bsdf(const MaterialEvalContext &ctx,
                                       SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    DiffuseBxDF get_diffuse_bsdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

  private:
    const void *ptr;
    Type type;
};

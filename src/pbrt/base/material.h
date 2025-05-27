#pragma once

class CoatedConductorMaterial;
class CoatedDiffuseMaterial;
class ConductorMaterial;
class DielectricMaterial;
class DiffuseMaterial;
class DiffuseTransmissionMaterial;
class MixMaterial;

class GPUMemoryAllocator;
class ParameterDictionary;
class SpectrumTexture;

class Material {
  public:
    enum class Type {
        coated_conductor,
        coated_diffuse,
        conductor,
        diffuse,
        diffuse_transmission,
        dielectric,
        mix,
    };

    static std::vector<Type> get_basic_material_types() {
        // consider only directly evaluable material for wavefront path tracing (excluding mix)
        return {
            Type::coated_conductor, Type::coated_diffuse, Type::conductor,
            Type::dielectric,       Type::diffuse,        Type::diffuse_transmission,
        };
    }

    static std::string material_type_to_string(Type type);

    static const Material *create(const std::string &type_of_material,
                                  const ParameterDictionary &parameters,
                                  GPUMemoryAllocator &allocator);

    static const Material *create_diffuse_material(const SpectrumTexture *texture,
                                                   GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Type get_material_type() const {
        return type;
    }

    PBRT_CPU_GPU
    const Material *get_material_from_mix_material(const Real u) const;

    PBRT_CPU_GPU
    BxDF get_bxdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

  private:
    Type type;
    const void *ptr = nullptr;

    void init(const CoatedConductorMaterial *coated_conductor_material);

    void init(const CoatedDiffuseMaterial *coated_diffuse_material);

    void init(const ConductorMaterial *conductor_material);

    void init(const DielectricMaterial *dielectric_material);

    void init(const DiffuseMaterial *diffuse_material);

    void init(const DiffuseTransmissionMaterial *diffuse_transmission_material);

    void init(const MixMaterial *mix_material);
};

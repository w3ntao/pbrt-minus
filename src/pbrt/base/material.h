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

    explicit Material(const DiffuseMaterial *diffuse_material)
        : type(Type::diffuse), ptr(diffuse_material) {}

    explicit Material(const CoatedConductorMaterial *coated_conductor_material)
        : type(Type::coated_conductor), ptr(coated_conductor_material) {}

    explicit Material(const CoatedDiffuseMaterial *coated_diffuse_material)
        : type(Type::coated_diffuse), ptr(coated_diffuse_material) {}

    explicit Material(const ConductorMaterial *conductor_material)
        : type(Type::conductor), ptr(conductor_material) {}

    explicit Material(const DielectricMaterial *dielectric_material)
        : type(Type::dielectric), ptr(dielectric_material) {}

    explicit Material(const DiffuseTransmissionMaterial *diffuse_transmission_material)
        : type(Type::diffuse_transmission), ptr(diffuse_transmission_material) {}

    explicit Material(const MixMaterial *mix_material) : type(Type::mix), ptr(mix_material) {}

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
    const Material *get_material_from_mix_material(Real u) const;

    PBRT_CPU_GPU
    BxDF get_bxdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

  private:
    Type type;
    const void *ptr = nullptr;
};

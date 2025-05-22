#pragma once

#include <cuda/std/variant>
#include <pbrt/materials/coated_conductor_material.h>
#include <pbrt/materials/coated_diffuse_material.h>
#include <pbrt/materials/conductor_material.h>
#include <pbrt/materials/dielectric_material.h>
#include <pbrt/materials/diffuse_material.h>
#include <pbrt/materials/diffuse_transmission_material.h>
#include <pbrt/materials/mix_material.h>

class GPUMemoryAllocator;
class ParameterDictionary;

namespace HIDDEN {
using MaterialVariants = cuda::std::variant<CoatedConductorMaterial, CoatedDiffuseMaterial,
                                            ConductorMaterial, DielectricMaterial, DiffuseMaterial,
                                            DiffuseTransmissionMaterial, MixMaterial>;
}

class Material : public HIDDEN::MaterialVariants {
    using HIDDEN::MaterialVariants::MaterialVariants;

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
        if (is_of_type<CoatedConductorMaterial>()) {
            return Type::coated_conductor;
        }

        if (is_of_type<CoatedDiffuseMaterial>()) {
            return Type::coated_diffuse;
        }

        if (is_of_type<ConductorMaterial>()) {
            return Type::conductor;
        }

        if (is_of_type<DiffuseMaterial>()) {
            return Type::diffuse;
        }

        if (is_of_type<DiffuseTransmissionMaterial>()) {
            return Type::diffuse_transmission;
        }

        if (is_of_type<DielectricMaterial>()) {
            return Type::dielectric;
        }

        if (is_of_type<MixMaterial>()) {
            return Type::mix;
        }

        REPORT_FATAL_ERROR();
    }

    PBRT_CPU_GPU
    const Material *get_mix_material(const SurfaceInteraction *si) const;

    PBRT_CPU_GPU
    BxDF get_bxdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const {
        return cuda::std::visit([&](auto &x) { return x.get_bxdf(ctx, lambda); }, *this);
    }

  private:
    template <typename MaterialType>
    PBRT_CPU_GPU bool is_of_type() const {
        const auto variant_ptr = static_cast<const HIDDEN::MaterialVariants *>(this);
        return cuda::std::holds_alternative<MaterialType>(*variant_ptr);
    }

    template <typename MaterialType>
    PBRT_CPU_GPU MaterialType convert() const {
        return cuda::std::get<MaterialType>(*this);
    }
};

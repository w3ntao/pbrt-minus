#include <map>
#include <pbrt/base/interaction.h>
#include <pbrt/base/material.h>
#include <pbrt/bxdfs/coated_conductor_bxdf.h>
#include <pbrt/bxdfs/conductor_bxdf.h>
#include <pbrt/bxdfs/dielectric_bxdf.h>
#include <pbrt/bxdfs/diffuse_bxdf.h>
#include <pbrt/bxdfs/diffuse_transmission_bxdf.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/materials/coated_conductor_material.h>
#include <pbrt/materials/coated_diffuse_material.h>
#include <pbrt/materials/conductor_material.h>
#include <pbrt/materials/dielectric_material.h>
#include <pbrt/materials/diffuse_material.h>
#include <pbrt/materials/diffuse_transmission_material.h>
#include <pbrt/materials/mix_material.h>

std::string Material::material_type_to_string(const Type type) {
    const std::map<Type, std::string> material_names = {
        {Type::coated_conductor, "CoatedConductor"},
        {Type::coated_diffuse, "CoatedDiffuse"},
        {Type::conductor, "Conductor"},
        {Type::dielectric, "Dielectric"},
        {Type::diffuse, "Diffuse"},
        {Type::diffuse_transmission, "DiffuseTransmission"},
        {Type::mix, "Mix"},
    };

    if (material_names.find(type) == material_names.end()) {
        REPORT_FATAL_ERROR();
    }

    return material_names.at(type);
}

const Material *Material::create(const std::string &type_of_material,
                                 const ParameterDictionary &parameters,
                                 GPUMemoryAllocator &allocator) {
    auto material = allocator.allocate<Material>();

    if (type_of_material == "coatedconductor") {
        auto coated_conductor_material = allocator.allocate<CoatedConductorMaterial>();

        coated_conductor_material->init(parameters, allocator);
        material->init(coated_conductor_material);

        return material;
    }

    if (type_of_material == "coateddiffuse") {
        auto coated_diffuse_material = allocator.allocate<CoatedDiffuseMaterial>();

        coated_diffuse_material->init(parameters, allocator);
        material->init(coated_diffuse_material);

        return material;
    }

    if (type_of_material == "conductor") {
        auto conductor_material = allocator.allocate<ConductorMaterial>();

        conductor_material->init(parameters, allocator);
        material->init(conductor_material);

        return material;
    }

    if (type_of_material == "dielectric") {
        auto dielectric_material = allocator.allocate<DielectricMaterial>();

        dielectric_material->init(parameters, allocator);
        material->init(dielectric_material);

        return material;
    }

    if (type_of_material == "diffuse") {
        auto diffuse_material = allocator.allocate<DiffuseMaterial>();

        diffuse_material->init(parameters, allocator);
        material->init(diffuse_material);

        return material;
    }

    if (type_of_material == "diffusetransmission") {
        const auto diffuse_transmission_material =
            DiffuseTransmissionMaterial::create(parameters, allocator);

        material->init(diffuse_transmission_material);

        return material;
    }

    if (type_of_material == "mix") {
        auto mix_material = allocator.allocate<MixMaterial>();

        mix_material->init(parameters);
        material->init(mix_material);

        return material;
    }

    printf("\nMaterial `%s` not implemented\n", type_of_material.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

const Material *Material::create_diffuse_material(const SpectrumTexture *texture,
                                                  GPUMemoryAllocator &allocator) {
    auto diffuse_material = DiffuseMaterial::create(texture, allocator);

    auto material = allocator.allocate<Material>();

    material->init(diffuse_material);

    return material;
}

void Material::init(const CoatedConductorMaterial *coated_conductor_material) {
    type = Type::coated_conductor;
    ptr = coated_conductor_material;
}

void Material::init(const CoatedDiffuseMaterial *coated_diffuse_material) {
    type = Type::coated_diffuse;
    ptr = coated_diffuse_material;
}

void Material::init(const ConductorMaterial *conductor_material) {
    type = Type::conductor;
    ptr = conductor_material;
}

void Material::init(const DielectricMaterial *dielectric_material) {
    type = Type::dielectric;
    ptr = dielectric_material;
}

void Material::init(const DiffuseMaterial *diffuse_material) {
    type = Type::diffuse;
    ptr = diffuse_material;
}

void Material::init(const DiffuseTransmissionMaterial *diffuse_transmission_material) {
    type = Type::diffuse_transmission;
    ptr = diffuse_transmission_material;
}

void Material::init(const MixMaterial *mix_material) {
    type = Type::mix;
    ptr = mix_material;
}

PBRT_CPU_GPU
const Material *Material::get_mix_material(const SurfaceInteraction *si) const {
    if (type != Type::mix) {
        REPORT_FATAL_ERROR();
    }

    return static_cast<const MixMaterial *>(ptr)->get_material(si);
}

PBRT_CPU_GPU
CoatedConductorBxDF Material::get_coated_conductor_bsdf(const MaterialEvalContext &ctx,
                                                        SampledWavelengths &lambda) const {
    if (type != Type::coated_conductor) {
        REPORT_FATAL_ERROR();
    }

    return static_cast<const CoatedConductorMaterial *>(ptr)->get_coated_conductor_bsdf(ctx,
                                                                                        lambda);
}

PBRT_CPU_GPU
CoatedDiffuseBxDF Material::get_coated_diffuse_bsdf(const MaterialEvalContext &ctx,
                                                    SampledWavelengths &lambda) const {
    if (type != Type::coated_diffuse) {
        REPORT_FATAL_ERROR();
    }

    return static_cast<const CoatedDiffuseMaterial *>(ptr)->get_coated_diffuse_bsdf(ctx, lambda);
}

PBRT_CPU_GPU
ConductorBxDF Material::get_conductor_bsdf(const MaterialEvalContext &ctx,
                                           SampledWavelengths &lambda) const {
    if (type != Type::conductor) {
        REPORT_FATAL_ERROR();
    }

    return static_cast<const ConductorMaterial *>(ptr)->get_conductor_bsdf(ctx, lambda);
}

PBRT_CPU_GPU
DielectricBxDF Material::get_dielectric_bsdf(const MaterialEvalContext &ctx,
                                             SampledWavelengths &lambda) const {
    if (type != Type::dielectric) {
        REPORT_FATAL_ERROR();
    }

    return static_cast<const DielectricMaterial *>(ptr)->get_dielectric_bsdf(ctx, lambda);
}

PBRT_CPU_GPU
DiffuseBxDF Material::get_diffuse_bsdf(const MaterialEvalContext &ctx,
                                       SampledWavelengths &lambda) const {
    if (type != Type::diffuse) {
        REPORT_FATAL_ERROR();
    }

    return static_cast<const DiffuseMaterial *>(ptr)->get_diffuse_bsdf(ctx, lambda);
}

PBRT_CPU_GPU
DiffuseTransmissionBxDF Material::get_diffuse_transmission_bsdf(const MaterialEvalContext &ctx,
                                                                SampledWavelengths &lambda) const {
    if (type != Type::diffuse_transmission) {
        REPORT_FATAL_ERROR();
    }

    return static_cast<const DiffuseTransmissionMaterial *>(ptr)->get_diffuse_transmission_bsdf(
        ctx, lambda);
}

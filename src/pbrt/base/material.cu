#include <map>
#include <pbrt/base/interaction.h>
#include <pbrt/base/material.h>
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
        auto coated_conductor_material = CoatedConductorMaterial::create(parameters, allocator);
        material->init(coated_conductor_material);

        return material;
    }

    if (type_of_material == "coateddiffuse") {
        auto coated_diffuse_material = CoatedDiffuseMaterial::create(parameters, allocator);
        material->init(coated_diffuse_material);

        return material;
    }

    if (type_of_material == "conductor") {
        auto conductor_material = ConductorMaterial::create(parameters, allocator);
        material->init(conductor_material);

        return material;
    }

    if (type_of_material == "dielectric") {
        auto dielectric_material = DielectricMaterial::create(parameters, allocator);
        material->init(dielectric_material);

        return material;
    }

    if (type_of_material == "diffuse") {
        auto diffuse_material = DiffuseMaterial::create(parameters, allocator);
        material->init(diffuse_material);

        return material;
    }

    if (type_of_material == "diffusetransmission") {
        auto diffuse_transmission_material =
            DiffuseTransmissionMaterial::create(parameters, allocator);
        material->init(diffuse_transmission_material);

        return material;
    }

    if (type_of_material == "mix") {
        auto mix_material = MixMaterial::create(parameters, allocator);
        material->init(mix_material);

        return material;
    }

    printf("\nMaterial `%s` not implemented\n", type_of_material.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

const Material *Material::create_diffuse_material(const SpectrumTexture *texture,
                                                  GPUMemoryAllocator &allocator) {
    auto material = allocator.allocate<Material>();

    auto diffuse_material = DiffuseMaterial::create(texture, allocator);
    material->init(diffuse_material);

    return material;
}

PBRT_CPU_GPU
const Material *Material::get_material_from_mix_material(const Real u) const {
    if (type != Type::mix) {
        REPORT_FATAL_ERROR();
    }

    return static_cast<const MixMaterial *>(ptr)->get_material(u);
}

PBRT_CPU_GPU
BxDF Material::get_bxdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const {
    switch (type) {
    case Type::coated_conductor: {
        return static_cast<const CoatedConductorMaterial *>(ptr)->get_bxdf(ctx, lambda);
    }

    case Type::coated_diffuse: {
        return static_cast<const CoatedDiffuseMaterial *>(ptr)->get_bxdf(ctx, lambda);
    }

    case Type::conductor: {
        return static_cast<const ConductorMaterial *>(ptr)->get_bxdf(ctx, lambda);
    }

    case Type::dielectric: {
        return static_cast<const DielectricMaterial *>(ptr)->get_bxdf(ctx, lambda);
    }

    case Type::diffuse: {
        return static_cast<const DiffuseMaterial *>(ptr)->get_bxdf(ctx, lambda);
    }

    case Type::diffuse_transmission: {
        return static_cast<const DiffuseTransmissionMaterial *>(ptr)->get_bxdf(ctx, lambda);
    }

    case Type::mix: {
        return static_cast<const MixMaterial *>(ptr)->get_bxdf(ctx, lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
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

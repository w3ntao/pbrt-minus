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
    if (type_of_material == "interface") {
        return nullptr;
    }

    if (type_of_material == "coatedconductor") {
        auto coated_conductor_material =
            allocator.create<CoatedConductorMaterial>(parameters, allocator);
        return allocator.create<Material>(coated_conductor_material);
    }

    if (type_of_material == "coateddiffuse") {
        auto coated_diffuse_material =
            allocator.create<CoatedDiffuseMaterial>(parameters, allocator);
        return allocator.create<Material>(coated_diffuse_material);
    }

    if (type_of_material == "conductor") {
        auto conductor_material = allocator.create<ConductorMaterial>(parameters, allocator);
        return allocator.create<Material>(conductor_material);
    }

    if (type_of_material == "dielectric") {
        auto dielectric_material = allocator.create<DielectricMaterial>(parameters, allocator);
        return allocator.create<Material>(dielectric_material);
    }

    if (type_of_material == "diffuse") {
        auto diffuse_material = allocator.create<DiffuseMaterial>(parameters, allocator);
        return allocator.create<Material>(diffuse_material);
    }

    if (type_of_material == "diffusetransmission") {
        auto diffuse_transmission_material =
            allocator.create<DiffuseTransmissionMaterial>(parameters, allocator);
        return allocator.create<Material>(diffuse_transmission_material);
    }

    if (type_of_material == "mix") {
        auto mix_material = allocator.create<MixMaterial>(parameters);
        return allocator.create<Material>(mix_material);
    }

    printf("\nMaterial `%s` not implemented\n", type_of_material.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

const Material *Material::create_diffuse_material(const SpectrumTexture *texture,
                                                  GPUMemoryAllocator &allocator) {
    auto diffuse_material = allocator.create<DiffuseMaterial>(texture);
    return allocator.create<Material>(diffuse_material);
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

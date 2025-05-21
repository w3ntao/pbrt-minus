#include <map>
#include <pbrt/base/interaction.h>
#include <pbrt/base/material.h>
#include <pbrt/gpu/gpu_memory_allocator.h>

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
        *material = CoatedConductorMaterial(parameters, allocator);
        return material;
    }

    if (type_of_material == "coateddiffuse") {
        *material = CoatedDiffuseMaterial(parameters, allocator);
        return material;
    }

    if (type_of_material == "conductor") {
        *material = ConductorMaterial(parameters, allocator);
        return material;
    }

    if (type_of_material == "dielectric") {
        *material = DielectricMaterial(parameters, allocator);
        return material;
    }

    if (type_of_material == "diffuse") {
        *material = DiffuseMaterial(parameters, allocator);
        return material;
    }

    if (type_of_material == "diffusetransmission") {
        *material = DiffuseTransmissionMaterial(parameters, allocator);
        return material;
    }

    if (type_of_material == "mix") {
        *material = MixMaterial(parameters);
        return material;
    }

    printf("\nMaterial `%s` not implemented\n", type_of_material.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

const Material *Material::create_diffuse_material(const SpectrumTexture *texture,
                                                  GPUMemoryAllocator &allocator) {
    auto material = allocator.allocate<Material>();
    *material = DiffuseMaterial(texture);
    return material;
}

PBRT_CPU_GPU
const Material *Material::get_mix_material(const SurfaceInteraction *si) const {
    if (!is_of_type<MixMaterial>()) {
        REPORT_FATAL_ERROR();
    }

    return convert<MixMaterial>().get_material(si);
}

// TODO: rewrite those ugly get_x_bsdf()
PBRT_CPU_GPU
CoatedConductorBxDF Material::get_coated_conductor_bsdf(const MaterialEvalContext &ctx,
                                                        SampledWavelengths &lambda) const {
    if (!is_of_type<CoatedConductorMaterial>()) {
        REPORT_FATAL_ERROR();
    }

    return convert<CoatedConductorMaterial>().get_coated_conductor_bsdf(ctx, lambda);
}

PBRT_CPU_GPU
CoatedDiffuseBxDF Material::get_coated_diffuse_bsdf(const MaterialEvalContext &ctx,
                                                    SampledWavelengths &lambda) const {
    if (!is_of_type<CoatedDiffuseMaterial>()) {
        REPORT_FATAL_ERROR();
    }

    return convert<CoatedDiffuseMaterial>().get_coated_diffuse_bsdf(ctx, lambda);
}

PBRT_CPU_GPU
ConductorBxDF Material::get_conductor_bsdf(const MaterialEvalContext &ctx,
                                           SampledWavelengths &lambda) const {
    if (!is_of_type<ConductorMaterial>()) {
        REPORT_FATAL_ERROR();
    }

    return convert<ConductorMaterial>().get_conductor_bsdf(ctx, lambda);
}

PBRT_CPU_GPU
DielectricBxDF Material::get_dielectric_bsdf(const MaterialEvalContext &ctx,
                                             SampledWavelengths &lambda) const {
    if (!is_of_type<DielectricMaterial>()) {
        REPORT_FATAL_ERROR();
    }

    return convert<DielectricMaterial>().get_dielectric_bsdf(ctx, lambda);
}

PBRT_CPU_GPU
DiffuseBxDF Material::get_diffuse_bsdf(const MaterialEvalContext &ctx,
                                       SampledWavelengths &lambda) const {
    if (!is_of_type<DiffuseMaterial>()) {
        REPORT_FATAL_ERROR();
    }

    return convert<DiffuseMaterial>().get_diffuse_bsdf(ctx, lambda);
}

PBRT_CPU_GPU
DiffuseTransmissionBxDF Material::get_diffuse_transmission_bsdf(const MaterialEvalContext &ctx,
                                                                SampledWavelengths &lambda) const {
    if (!is_of_type<DiffuseTransmissionMaterial>()) {
        REPORT_FATAL_ERROR();
    }

    return convert<DiffuseTransmissionMaterial>().get_diffuse_transmission_bsdf(ctx, lambda);
}

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
const Material *Material::get_material_from_mix_material(const Real u) const {
    return cuda::std::get<MixMaterial>(*this).get_material(u);
}

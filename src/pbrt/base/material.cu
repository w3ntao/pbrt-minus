#include "pbrt/base/interaction.h"
#include "pbrt/base/material.h"
#include "pbrt/bxdfs/coated_conductor_bxdf.h"
#include "pbrt/bxdfs/conductor_bxdf.h"
#include "pbrt/bxdfs/dielectric_bxdf.h"
#include "pbrt/bxdfs/diffuse_bxdf.h"
#include "pbrt/materials/coated_conductor_material.h"
#include "pbrt/materials/coated_diffuse_material.h"
#include "pbrt/materials/conductor_material.h"
#include "pbrt/materials/dielectric_material.h"
#include "pbrt/materials/diffuse_material.h"
#include "pbrt/materials/mix_material.h"

const Material *Material::create(const std::string &type_of_material,
                                 const ParameterDictionary &parameters,
                                 std::vector<void *> &gpu_dynamic_pointers) {
    if (type_of_material == "coatedconductor") {
        CoatedConductorMaterial *coated_conductor_material;
        Material *material;
        CHECK_CUDA_ERROR(
            cudaMallocManaged(&coated_conductor_material, sizeof(CoatedConductorMaterial)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&material, sizeof(Material)));

        gpu_dynamic_pointers.push_back(coated_conductor_material);
        gpu_dynamic_pointers.push_back(material);

        coated_conductor_material->init(parameters, gpu_dynamic_pointers);
        material->init(coated_conductor_material);

        return material;
    }

    if (type_of_material == "coateddiffuse") {
        CoatedDiffuseMaterial *coated_diffuse_material;
        Material *material;
        CHECK_CUDA_ERROR(
            cudaMallocManaged(&coated_diffuse_material, sizeof(CoatedDiffuseMaterial)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&material, sizeof(Material)));

        coated_diffuse_material->init(parameters, gpu_dynamic_pointers);
        material->init(coated_diffuse_material);

        gpu_dynamic_pointers.push_back(coated_diffuse_material);
        gpu_dynamic_pointers.push_back(material);

        return material;
    }

    if (type_of_material == "conductor") {
        ConductorMaterial *conductor_material;
        Material *material;
        CHECK_CUDA_ERROR(cudaMallocManaged(&conductor_material, sizeof(ConductorMaterial)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&material, sizeof(Material)));

        gpu_dynamic_pointers.push_back(conductor_material);
        gpu_dynamic_pointers.push_back(material);

        conductor_material->init(parameters, gpu_dynamic_pointers);
        material->init(conductor_material);

        return material;
    }

    if (type_of_material == "dielectric") {
        DielectricMaterial *dielectric_material;
        Material *material;

        CHECK_CUDA_ERROR(cudaMallocManaged(&dielectric_material, sizeof(DielectricMaterial)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&material, sizeof(Material)));

        gpu_dynamic_pointers.push_back(dielectric_material);
        gpu_dynamic_pointers.push_back(material);

        dielectric_material->init(parameters, gpu_dynamic_pointers);
        material->init(dielectric_material);

        return material;
    }

    if (type_of_material == "diffuse") {
        DiffuseMaterial *diffuse_material;
        Material *material;
        CHECK_CUDA_ERROR(cudaMallocManaged(&diffuse_material, sizeof(DiffuseMaterial)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&material, sizeof(Material)));

        gpu_dynamic_pointers.push_back(diffuse_material);
        gpu_dynamic_pointers.push_back(material);

        diffuse_material->init(parameters, gpu_dynamic_pointers);
        material->init(diffuse_material);

        return material;
    }

    if (type_of_material == "mix") {
        MixMaterial *mix_material;
        Material *material;
        CHECK_CUDA_ERROR(cudaMallocManaged(&mix_material, sizeof(MixMaterial)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&material, sizeof(Material)));

        gpu_dynamic_pointers.push_back(mix_material);
        gpu_dynamic_pointers.push_back(material);

        mix_material->init(parameters);
        material->init(mix_material);

        return material;
    }

    printf("\nMaterial `%s` not implemented\n", type_of_material.c_str());

    REPORT_FATAL_ERROR();
    return nullptr;
}

const Material *Material::create_diffuse_material(const SpectrumTexture *texture,
                                                  std::vector<void *> &gpu_dynamic_pointers) {
    auto diffuse_material = DiffuseMaterial::create(texture, gpu_dynamic_pointers);

    Material *material;
    CHECK_CUDA_ERROR(cudaMallocManaged(&material, sizeof(Material)));
    gpu_dynamic_pointers.push_back(material);

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

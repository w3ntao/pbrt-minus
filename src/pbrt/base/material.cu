#include "pbrt/base/material.h"

#include "pbrt/bxdfs/conductor_bxdf.h"
#include "pbrt/bxdfs/diffuse_bxdf.h"
#include "pbrt/bxdfs/dielectric_bxdf.h"

#include "pbrt/materials/coated_diffuse_material.h"
#include "pbrt/materials/conductor_material.h"
#include "pbrt/materials/diffuse_material.h"
#include "pbrt/materials/dielectric_material.h"

const Material *
Material::create_coated_diffuse_material(const ParameterDict &parameters,
                                         std::vector<void *> &gpu_dynamic_pointers) {
    CoatedDiffuseMaterial *coated_diffuse_material;
    Material *material;
    CHECK_CUDA_ERROR(cudaMallocManaged(&coated_diffuse_material, sizeof(CoatedDiffuseMaterial)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&material, sizeof(Material)));

    coated_diffuse_material->init(parameters, gpu_dynamic_pointers);
    material->init(coated_diffuse_material);

    gpu_dynamic_pointers.push_back(coated_diffuse_material);
    gpu_dynamic_pointers.push_back(material);

    return material;
}

const Material *Material::create_conductor_material(const ParameterDict &parameters,
                                                    std::vector<void *> &gpu_dynamic_pointers) {
    ConductorMaterial *conductor_material;
    CHECK_CUDA_ERROR(cudaMallocManaged(&conductor_material, sizeof(ConductorMaterial)));
    conductor_material->init(parameters, gpu_dynamic_pointers);

    Material *material;
    CHECK_CUDA_ERROR(cudaMallocManaged(&material, sizeof(Material)));
    material->init(conductor_material);

    gpu_dynamic_pointers.push_back(conductor_material);
    gpu_dynamic_pointers.push_back(material);

    return material;
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

PBRT_GPU
CoatedDiffuseBxDF Material::get_coated_diffuse_bsdf(const MaterialEvalContext &ctx,
                                                    SampledWavelengths &lambda) const {
    if (type != Type::coated_diffuse) {
        REPORT_FATAL_ERROR();
    }

    return ((CoatedDiffuseMaterial *)ptr)->get_coated_diffuse_bsdf(ctx, lambda);
}

PBRT_GPU
ConductorBxDF Material::get_conductor_bsdf(const MaterialEvalContext &ctx,
                                           SampledWavelengths &lambda) const {
    if (type != Type::conductor) {
        REPORT_FATAL_ERROR();
    }

    return ((ConductorMaterial *)ptr)->get_conductor_bsdf(ctx, lambda);
}

PBRT_GPU
DielectricBxDF Material::get_dielectric_bsdf(const MaterialEvalContext &ctx,
                                             SampledWavelengths &lambda) const {
    if (type != Type::dielectric) {
        REPORT_FATAL_ERROR();
    }

    return ((DielectricMaterial *)ptr)->get_dielectric_bsdf(ctx, lambda);
}

PBRT_GPU
DiffuseBxDF Material::get_diffuse_bsdf(const MaterialEvalContext &ctx,
                                       SampledWavelengths &lambda) const {
    if (type != Type::diffuse) {
        REPORT_FATAL_ERROR();
    }

    return ((DiffuseMaterial *)ptr)->get_diffuse_bsdf(ctx, lambda);
}

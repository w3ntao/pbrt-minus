#include "pbrt/base/material.h"

#include "pbrt/bxdfs/diffuse_bxdf.h"
#include "pbrt/bxdfs/dielectric_bxdf.h"

#include "pbrt/materials/diffuse_material.h"
#include "pbrt/materials/dielectric_material.h"
#include "pbrt/materials/coated_diffuse_material.h"

const Material *Material::create_diffuse_material(const SpectrumTexture *texture,
                                                  std::vector<void *> &gpu_dynamic_pointers) {
    auto diffuse_material = DiffuseMaterial::create(texture, gpu_dynamic_pointers);

    Material *material;
    CHECK_CUDA_ERROR(cudaMallocManaged(&material, sizeof(Material)));
    material->init(diffuse_material);

    gpu_dynamic_pointers.push_back(material);
    return material;
}

void Material::init(const DiffuseMaterial *diffuse_material) {
    type = Type::diffuse;
    ptr = diffuse_material;
}

void Material::init(const DielectricMaterial *dielectric_material) {
    type = Type::dielectric;
    ptr = dielectric_material;
}

void Material::init(const CoatedDiffuseMaterial *coated_diffuse_material) {
    type = Type::coated_diffuse;
    ptr = coated_diffuse_material;
}

PBRT_GPU
DiffuseBxDF Material::get_diffuse_bsdf(const MaterialEvalContext &ctx,
                                       SampledWavelengths &lambda) const {
    if (type != Type::diffuse) {
        REPORT_FATAL_ERROR();
    }

    return ((DiffuseMaterial *)ptr)->get_diffuse_bsdf(ctx, lambda);
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
CoatedDiffuseBxDF Material::get_coated_diffuse_bsdf(const MaterialEvalContext &ctx,
                                                    SampledWavelengths &lambda) const {
    if (type != Type::coated_diffuse) {
        REPORT_FATAL_ERROR();
    }

    return ((CoatedDiffuseMaterial *)ptr)->get_coated_diffuse_bsdf(ctx, lambda);
}

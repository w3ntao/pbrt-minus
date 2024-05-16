#include "pbrt/base/material.h"

#include "pbrt/bxdfs/diffuse_bxdf.h"
#include "pbrt/bxdfs/dielectric_bxdf.h"

#include "pbrt/materials/diffuse_material.h"
#include "pbrt/materials/dielectric_material.h"

void Material::init(const DiffuseMaterial *diffuse_material) {
    type = Type::diffuse;
    ptr = diffuse_material;
}

void Material::init(const DielectricMaterial *dielectric_material) {
    type = Type::dieletric;
    ptr = dielectric_material;
}

PBRT_GPU
DiffuseBxDF Material::get_diffuse_bsdf(const MaterialEvalContext &ctx,
                                       SampledWavelengths &lambda) const {
    if (type == Type::diffuse) {
        return ((DiffuseMaterial *)ptr)->get_diffuse_bsdf(ctx, lambda);
    }
    
    REPORT_FATAL_ERROR();
    return {};
}

PBRT_GPU
DielectricBxDF Material::get_dielectric_bsdf(const MaterialEvalContext &ctx,
                                             SampledWavelengths &lambda) const {
    if (type == Type::dieletric) {
        return ((DielectricMaterial *)ptr)->get_dielectric_bsdf(ctx, lambda);
    }

    REPORT_FATAL_ERROR();
    return {};
}

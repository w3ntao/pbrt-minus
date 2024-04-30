#include "pbrt/base/material.h"
#include "pbrt/materials/diffuse_material.h"
#include "pbrt/bxdfs/diffuse_bxdf.h"

void Material::init(const DiffuseMaterial *diffuse_material) {
    material_ptr = (void *)diffuse_material;
    material_type = MaterialType::diffuse_material;
}

PBRT_GPU
DiffuseBxDF Material::get_diffuse_bsdf(const MaterialEvalContext &ctx,
                                       SampledWavelengths &lambda) const {
    switch (material_type) {
    case (MaterialType::diffuse_material): {
        return ((DiffuseMaterial *)material_ptr)->get_diffuse_bsdf(ctx, lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

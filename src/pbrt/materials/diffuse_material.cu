#include "pbrt/materials/diffuse_material.h"

#include "pbrt/base/material.h"
#include "pbrt/bxdfs/diffuse_bxdf.h"
#include "pbrt/textures/spectrum_constant_texture.h"
#include "pbrt/spectrum_util/sampled_wavelengths.h"

void DiffuseMaterial::init(const SpectrumConstantTexture *_reflectance) {
    reflectance = _reflectance;
}

PBRT_GPU
void DiffuseMaterial::get_diffuse_bsdf(BSDF &bsdf, const MaterialEvalContext &ctx,
                                       SampledWavelengths &lambda) const {
    auto r = reflectance->evaluate(ctx, lambda).clamp(0.0, 1.0);
    bsdf.diffuse_bxdf.init(r);
}

#include "pbrt/materials/diffuse_material.h"

#include "pbrt/base/material.h"
#include "pbrt/base/texture.h"
#include "pbrt/bxdfs/diffuse_bxdf.h"
#include "pbrt/spectrum_util/sampled_wavelengths.h"

void DiffuseMaterial::init(const SpectrumTexture *_reflectance) {
    reflectance = _reflectance;
}

PBRT_GPU
DiffuseBxDF DiffuseMaterial::get_diffuse_bsdf(const MaterialEvalContext &ctx,
                                              SampledWavelengths &lambda) const {
    auto r = reflectance->evaluate(ctx, lambda).clamp(0.0, 1.0);
    return DiffuseBxDF(r);
}

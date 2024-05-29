#include "pbrt/materials/dielectric_material.h"

#include "pbrt/base/material.h"
#include "pbrt/base/spectrum.h"
#include "pbrt/base/texture.h"

#include "pbrt/bxdfs/dielectric_bxdf.h"
#include "pbrt/spectra/constant_spectrum.h"
#include "pbrt/scene/parameter_dict.h"
#include "pbrt/textures/float_constant_texture.h"

void DielectricMaterial::init(const ParameterDict &parameters,
                              std::vector<void *> &gpu_dynamic_pointers) {
    if (parameters.has_floats("eta")) {
        // this part is not implemented
        REPORT_FATAL_ERROR();

    } else {
        // TODO: if eta was initialized as a Spectrum?
        eta = Spectrum::create_constant_spectrum(1.5, gpu_dynamic_pointers);
    }

    auto float_u_roughness = parameters.get_float("uroughness", 0.0);
    auto float_v_roughness = parameters.get_float("uroughness", 0.0);

    u_roughness = FloatTexture::create(float_u_roughness, gpu_dynamic_pointers);
    v_roughness = FloatTexture::create(float_v_roughness, gpu_dynamic_pointers);

    remap_roughness = parameters.get_bool("remaproughness", true);
}

PBRT_GPU
DielectricBxDF DielectricMaterial::get_dielectric_bsdf(const MaterialEvalContext &ctx,
                                                       SampledWavelengths &lambda) const {
    // Compute index of refraction for dielectric material
    FloatType sampled_eta = (*eta)(lambda[0]);
    if (!eta->is_constant_spectrum()) {
        lambda.TerminateSecondary();
    }

    // Handle edge case in case lambda[0] is beyond the wavelengths stored by the
    // Spectrum.
    if (sampled_eta == 0) {
        sampled_eta = 1;
    }

    // Create microfacet distribution for dielectric material

    auto urough = u_roughness->evaluate(ctx);
    auto vrough = v_roughness->evaluate(ctx);

    if (remap_roughness) {
        urough = TrowbridgeReitzDistribution::RoughnessToAlpha(urough);
        vrough = TrowbridgeReitzDistribution::RoughnessToAlpha(vrough);
    }
    TrowbridgeReitzDistribution distrib(urough, vrough);
    return DielectricBxDF(sampled_eta, distrib);
}

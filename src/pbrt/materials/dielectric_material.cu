#include "pbrt/materials/dielectric_material.h"

#include "pbrt/base/float_texture.h"
#include "pbrt/base/material.h"
#include "pbrt/base/spectrum.h"

#include "pbrt/bxdfs/dielectric_bxdf.h"
#include "pbrt/spectra/constant_spectrum.h"
#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/textures/float_constant_texture.h"

void DielectricMaterial::init(const ParameterDictionary &parameters,
                              std::vector<void *> &gpu_dynamic_pointers) {
    auto key_eta = "eta";

    eta = nullptr;

    if (parameters.has_floats(key_eta)) {
        // this part is not implemented
        REPORT_FATAL_ERROR();
    } else if (parameters.has_spectrum(key_eta)) {
        REPORT_FATAL_ERROR();
    } else {
        eta = Spectrum::create_constant_spectrum(1.5, gpu_dynamic_pointers);
    }

    auto uroughness_key = "uroughness";
    if (parameters.has_float_texture(uroughness_key)) {
        u_roughness = parameters.get_float_texture(uroughness_key);
    } else if (parameters.has_floats(uroughness_key)) {
        auto uroughness_val = parameters.get_float(uroughness_key, {});
        u_roughness =
            FloatTexture::create_constant_float_texture(uroughness_val, gpu_dynamic_pointers);
    } else {
        auto roughness_val = parameters.get_float("roughness", 0.0);
        u_roughness =
            FloatTexture::create_constant_float_texture(roughness_val, gpu_dynamic_pointers);
    }

    auto vroughness_key = "vroughness";
    if (parameters.has_float_texture(vroughness_key)) {
        v_roughness = parameters.get_float_texture(vroughness_key);
    } else if (parameters.has_floats(vroughness_key)) {
        auto vroughness_val = parameters.get_float(vroughness_key, {});
        v_roughness =
            FloatTexture::create_constant_float_texture(vroughness_val, gpu_dynamic_pointers);
    } else {
        auto roughness_val = parameters.get_float("roughness", 0.0);
        v_roughness =
            FloatTexture::create_constant_float_texture(roughness_val, gpu_dynamic_pointers);
    }

    remap_roughness = parameters.get_bool("remaproughness", true);

    if (eta == nullptr) {
        REPORT_FATAL_ERROR();
    }
}

PBRT_GPU
DielectricBxDF DielectricMaterial::get_dielectric_bsdf(const MaterialEvalContext &ctx,
                                                       SampledWavelengths &lambda) const {
    // Compute index of refraction for dielectric material
    FloatType sampled_eta = (*eta)(lambda[0]);
    if (!eta->is_constant_spectrum()) {
        lambda.terminate_secondary();
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

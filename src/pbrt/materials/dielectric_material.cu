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
    eta = nullptr;

    auto key_eta = "eta";
    if (parameters.has_floats(key_eta)) {
        eta =
            Spectrum::create_constant_spectrum(parameters.get_float(key_eta), gpu_dynamic_pointers);
    } else {
        eta = parameters.get_spectrum(key_eta, SpectrumType::Unbounded, gpu_dynamic_pointers);
    }

    if (!eta) {
        eta = Spectrum::create_constant_spectrum(1.5, gpu_dynamic_pointers);
    }

    uRoughness = parameters.get_float_texture_or_null("uroughness", gpu_dynamic_pointers);
    if (!uRoughness) {
        auto roughness_val = parameters.get_float("roughness", 0.0);
        uRoughness =
            FloatTexture::create_constant_float_texture(roughness_val, gpu_dynamic_pointers);
    }

    vRoughness = parameters.get_float_texture_or_null("vroughness", gpu_dynamic_pointers);
    if (!vRoughness) {
        auto roughness_val = parameters.get_float("roughness", 0.0);
        vRoughness =
            FloatTexture::create_constant_float_texture(roughness_val, gpu_dynamic_pointers);
    }

    remapRoughness = parameters.get_bool("remaproughness", true);

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

    auto urough = uRoughness->evaluate(ctx);
    auto vrough = vRoughness->evaluate(ctx);

    if (remapRoughness) {
        urough = TrowbridgeReitzDistribution::RoughnessToAlpha(urough);
        vrough = TrowbridgeReitzDistribution::RoughnessToAlpha(vrough);
    }
    TrowbridgeReitzDistribution distrib(urough, vrough);

    return DielectricBxDF(sampled_eta, distrib);
}

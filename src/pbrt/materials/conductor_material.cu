#include "pbrt/materials/conductor_material.h"

#include "pbrt/base/float_texture.h"
#include "pbrt/base/material.h"
#include "pbrt/base/spectrum.h"

#include "pbrt/bxdfs/conductor_bxdf.h"

#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"

#include "pbrt/spectrum_util/global_spectra.h"

void ConductorMaterial::init(const ParameterDictionary &parameters,
                             std::vector<void *> &gpu_dynamic_pointers) {
    auto key_eta = "eta";
    if (parameters.has_spectrum_texture(key_eta)) {
        REPORT_FATAL_ERROR();
    }

    auto spectrum_eta =
        parameters.get_spectrum(key_eta, SpectrumType::Albedo, gpu_dynamic_pointers);

    if (spectrum_eta == nullptr) {
        spectrum_eta =
            parameters.get_spectrum("metal-Cu-eta", SpectrumType::Albedo, gpu_dynamic_pointers);
    }
    eta = SpectrumTexture::create_constant_texture(spectrum_eta, gpu_dynamic_pointers);

    auto key_k = "k";
    if (parameters.has_spectrum_texture(key_k)) {
        REPORT_FATAL_ERROR();
    }

    auto spectrum_k = parameters.get_spectrum(key_k, SpectrumType::Albedo, gpu_dynamic_pointers);
    if (spectrum_k == nullptr) {
        REPORT_FATAL_ERROR();
    }
    k = SpectrumTexture::create_constant_texture(spectrum_k, gpu_dynamic_pointers);

    auto key_reflectance = "reflectance";
    if (parameters.has_spectrum_texture(key_reflectance)) {
        REPORT_FATAL_ERROR();
    } else if (parameters.has_spectrum(key_reflectance)) {
        auto spectrum_reflectance =
            parameters.get_spectrum(key_reflectance, SpectrumType::Albedo, gpu_dynamic_pointers);
        reflectance =
            SpectrumTexture::create_constant_texture(spectrum_reflectance, gpu_dynamic_pointers);
    } else {
        reflectance = nullptr;
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
}

PBRT_GPU
ConductorBxDF ConductorMaterial::get_conductor_bsdf(const MaterialEvalContext &ctx,
                                                    SampledWavelengths &lambda) {
    auto uRough = u_roughness->evaluate(ctx);
    auto vRough = v_roughness->evaluate(ctx);

    if (remap_roughness) {
        uRough = TrowbridgeReitzDistribution::RoughnessToAlpha(uRough);
        vRough = TrowbridgeReitzDistribution::RoughnessToAlpha(vRough);
    }

    SampledSpectrum etas, ks;
    if (eta) {
        etas = eta->evaluate(ctx, lambda);
        ks = k->evaluate(ctx, lambda);

    } else {
        // Avoid r==0 NaN case...
        auto r = reflectance->evaluate(ctx, lambda).clamp(0, 0.9999);
        etas = SampledSpectrum(1.f);
        ks = 2 * r.sqrt() / (SampledSpectrum(1) - r).clamp(0, Infinity).sqrt();
    }
    TrowbridgeReitzDistribution distrib(uRough, vRough);
    return ConductorBxDF(distrib, etas, ks);
}

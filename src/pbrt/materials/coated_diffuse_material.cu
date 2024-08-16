#include "pbrt/materials/coated_diffuse_material.h"

#include "pbrt/base/float_texture.h"
#include "pbrt/base/material.h"
#include "pbrt/base/spectrum.h"

#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/spectra/constant_spectrum.h"
#include "pbrt/spectrum_util/global_spectra.h"
#include "pbrt/textures/spectrum_constant_texture.h"

void CoatedDiffuseMaterial::init(const ParameterDictionary &parameters,
                                 std::vector<void *> &gpu_dynamic_pointers) {
    reflectance = nullptr;
    albedo = nullptr;

    u_roughness = nullptr;
    v_roughness = nullptr;
    thickness = nullptr;
    g = nullptr;

    eta = nullptr;

    reflectance =
        parameters.get_spectrum_texture("reflectance", SpectrumType::Albedo, gpu_dynamic_pointers);
    if (!reflectance) {
        reflectance = SpectrumTexture::create_constant_float_val_texture(0.5, gpu_dynamic_pointers);
    }

    u_roughness = parameters.get_float_texture_or_null("uroughness", gpu_dynamic_pointers);
    if (!u_roughness) {
        auto roughness_val = parameters.get_float("roughness", 0.0);
        u_roughness =
            FloatTexture::create_constant_float_texture(roughness_val, gpu_dynamic_pointers);
    }

    v_roughness = parameters.get_float_texture_or_null("vroughness", gpu_dynamic_pointers);
    if (!v_roughness) {
        auto roughness_val = parameters.get_float("roughness", 0.0);
        v_roughness =
            FloatTexture::create_constant_float_texture(roughness_val, gpu_dynamic_pointers);
    }

    thickness =
        parameters.get_float_texture_with_default_val("thickness", 0.01, gpu_dynamic_pointers);

    auto eta_val = parameters.get_float("eta", 1.5);
    eta = Spectrum::create_constant_spectrum(eta_val, gpu_dynamic_pointers);

    auto g_val = parameters.get_float("g", 0.0);
    g = FloatTexture::create_constant_float_texture(g_val, gpu_dynamic_pointers);

    auto albedo_val = parameters.get_float("albedo", 0.0);
    albedo = SpectrumTexture::create_constant_float_val_texture(albedo_val, gpu_dynamic_pointers);

    maxDepth = parameters.get_integer("maxdepth", 10);
    nSamples = parameters.get_integer("nsamples", 1);

    remapRoughness = parameters.get_bool("remaproughness", true);

    if (reflectance == nullptr || albedo == nullptr) {
        REPORT_FATAL_ERROR();
    }
    if (u_roughness == nullptr || v_roughness == nullptr || thickness == nullptr || g == nullptr) {
        REPORT_FATAL_ERROR();
    }
    if (eta == nullptr) {
        REPORT_FATAL_ERROR();
    }
}

PBRT_GPU
CoatedDiffuseBxDF CoatedDiffuseMaterial::get_coated_diffuse_bsdf(const MaterialEvalContext &ctx,
                                                                 SampledWavelengths &lambda) const {

    // Initialize diffuse component of plastic material
    SampledSpectrum r = reflectance->evaluate(ctx, lambda).clamp(0, 1);

    // Create microfacet distribution _distrib_ for coated diffuse material
    FloatType urough = u_roughness->evaluate(ctx);
    FloatType vrough = v_roughness->evaluate(ctx);

    if (remapRoughness) {
        urough = TrowbridgeReitzDistribution::RoughnessToAlpha(urough);
        vrough = TrowbridgeReitzDistribution::RoughnessToAlpha(vrough);
    }

    TrowbridgeReitzDistribution distrib(urough, vrough);

    FloatType thick = thickness->evaluate(ctx);

    FloatType sampledEta = (*eta)(lambda[0]);

    if (!eta->is_constant_spectrum()) {
        lambda.terminate_secondary();
    }
    if (sampledEta == 0) {
        sampledEta = 1;
    }

    SampledSpectrum a = albedo->evaluate(ctx, lambda).clamp(0, 1);
    FloatType gg = clamp<FloatType>(g->evaluate(ctx), -1, 1);

    return CoatedDiffuseBxDF(DielectricBxDF(sampledEta, distrib), DiffuseBxDF(r), thick, a, gg,
                             maxDepth, nSamples);
}

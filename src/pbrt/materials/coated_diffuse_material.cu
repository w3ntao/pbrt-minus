#include "pbrt/materials/coated_diffuse_material.h"

#include "pbrt/base/material.h"
#include "pbrt/base/texture.h"
#include "pbrt/base/spectrum.h"
#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/spectra/constant_spectrum.h"
#include "pbrt/textures/spectrum_constant_texture.h"

void CoatedDiffuseMaterial::init(const ParameterDictionary &parameters,
                                 std::vector<void *> &gpu_dynamic_pointers) {

    auto reflectance_key = "reflectance";
    if (parameters.has_spectrum_texture(reflectance_key)) {
        reflectance = parameters.get_spectrum_texture(reflectance_key);
    } else if (parameters.has_rgb(reflectance_key)) {
        auto rgb_val = parameters.get_rgb(reflectance_key, {});

        auto rgb_albedo_spectrum = Spectrum::create_rgb_albedo_spectrum(
            rgb_val, gpu_dynamic_pointers, parameters.rgb_color_space);

        reflectance =
            SpectrumTexture::create_constant_texture(rgb_albedo_spectrum, gpu_dynamic_pointers);
    } else {
        reflectance = SpectrumTexture::create_constant_float_val_texture(0.5, gpu_dynamic_pointers);
    }

    auto uroughness_key = "uroughness";
    if (parameters.has_float_texture(uroughness_key)) {
        u_roughness = parameters.get_float_texture(uroughness_key);
    } else if (parameters.has_floats(uroughness_key)) {
        auto uroughness_val = parameters.get_float(uroughness_key, {});
        u_roughness = FloatTexture::create(uroughness_val, gpu_dynamic_pointers);
    } else {
        auto roughness_val = parameters.get_float("roughness", 0.0);
        u_roughness = FloatTexture::create(roughness_val, gpu_dynamic_pointers);
    }

    auto vroughness_key = "vroughness";
    if (parameters.has_float_texture(vroughness_key)) {
        v_roughness = parameters.get_float_texture(vroughness_key);
    } else if (parameters.has_floats(vroughness_key)) {
        auto vroughness_val = parameters.get_float(vroughness_key, {});
        v_roughness = FloatTexture::create(vroughness_val, gpu_dynamic_pointers);
    } else {
        auto roughness_val = parameters.get_float("roughness", 0.0);
        v_roughness = FloatTexture::create(roughness_val, gpu_dynamic_pointers);
    }

    auto thickness_key = "thickness";
    if (parameters.has_float_texture(thickness_key)) {
        thickness = parameters.get_float_texture(thickness_key);
    } else {
        auto thickness_val = parameters.get_float(thickness_key, 0.01);
        thickness = FloatTexture::create(thickness_val, gpu_dynamic_pointers);
    }

    auto eta_val = parameters.get_float("eta", 1.5);
    eta = Spectrum::create_constant_spectrum(eta_val, gpu_dynamic_pointers);

    auto g_val = parameters.get_float("g", 0.0);
    g = FloatTexture::create(g_val, gpu_dynamic_pointers);

    auto albedo_val = parameters.get_float("albedo", 0.0);
    albedo = SpectrumTexture::create_constant_float_val_texture(albedo_val, gpu_dynamic_pointers);

    auto vec_max_depth = parameters.get_integer("maxdepth");
    maxDepth = vec_max_depth.empty() ? 10 : vec_max_depth[0];

    auto vec_n_samples = parameters.get_integer("nsamples");
    nSamples = vec_n_samples.empty() ? 1 : vec_n_samples[0];

    remapRoughness = parameters.get_bool("remaproughness", true);
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

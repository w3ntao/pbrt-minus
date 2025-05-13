#include <pbrt/base/float_texture.h>
#include <pbrt/base/material.h>
#include <pbrt/base/spectrum.h>
#include <pbrt/materials/coated_diffuse_material.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/textures/spectrum_constant_texture.h>

void CoatedDiffuseMaterial::init(const ParameterDictionary &parameters,
                                 GPUMemoryAllocator &allocator) {
    reflectance = nullptr;
    albedo = nullptr;

    u_roughness = nullptr;
    v_roughness = nullptr;
    thickness = nullptr;
    g = nullptr;

    eta = nullptr;

    reflectance = parameters.get_spectrum_texture("reflectance", SpectrumType::Albedo, allocator);
    if (!reflectance) {
        reflectance = SpectrumTexture::create_constant_float_val_texture(0.5, allocator);
    }

    u_roughness = parameters.get_float_texture_or_null("uroughness", allocator);
    if (!u_roughness) {
        u_roughness = parameters.get_float_texture("roughness", 0.0, allocator);
    }

    v_roughness = parameters.get_float_texture_or_null("vroughness", allocator);
    if (!v_roughness) {
        v_roughness = parameters.get_float_texture("roughness", 0.0, allocator);
    }

    thickness = parameters.get_float_texture_with_default_val("thickness", 0.01, allocator);

    auto eta_val = parameters.get_float("eta", 1.5);
    eta = Spectrum::create_constant_spectrum(eta_val, allocator);

    auto g_val = parameters.get_float("g", 0.0);
    g = FloatTexture::create_constant_float_texture(g_val, allocator);

    auto albedo_val = parameters.get_float("albedo", 0.0);
    albedo = SpectrumTexture::create_constant_float_val_texture(albedo_val, allocator);

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

PBRT_CPU_GPU
CoatedDiffuseBxDF CoatedDiffuseMaterial::get_coated_diffuse_bsdf(const MaterialEvalContext &ctx,
                                                                 SampledWavelengths &lambda) const {

    // Initialize diffuse component of plastic material
    SampledSpectrum r = reflectance->evaluate(ctx, lambda).clamp(0, 1);

    // Create microfacet distribution _distrib_ for coated diffuse material
    Real urough = u_roughness->evaluate(ctx);
    Real vrough = v_roughness->evaluate(ctx);

    if (remapRoughness) {
        urough = TrowbridgeReitzDistribution::RoughnessToAlpha(urough);
        vrough = TrowbridgeReitzDistribution::RoughnessToAlpha(vrough);
    }

    TrowbridgeReitzDistribution distrib(urough, vrough);

    Real thick = thickness->evaluate(ctx);

    Real sampledEta = (*eta)(lambda[0]);

    if (!eta->is_constant_spectrum()) {
        lambda.terminate_secondary();
    }
    if (sampledEta == 0) {
        sampledEta = 1;
    }

    SampledSpectrum a = albedo->evaluate(ctx, lambda).clamp(0, 1);
    Real gg = clamp<Real>(g->evaluate(ctx), -1, 1);

    return CoatedDiffuseBxDF(DielectricBxDF(sampledEta, distrib), DiffuseBxDF(r), thick, a, gg,
                             maxDepth, nSamples);
}

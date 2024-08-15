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
    eta = parameters.get_spectrum_texture("eta", SpectrumType::Unbounded, gpu_dynamic_pointers);
    k = parameters.get_spectrum_texture("k", SpectrumType::Unbounded, gpu_dynamic_pointers);
    reflectance =
        parameters.get_spectrum_texture("reflectance", SpectrumType::Albedo, gpu_dynamic_pointers);

    if (reflectance && (eta || k)) {
        printf("ERROR: for ConductorMaterial, both `reflectance` and (`eta` and `k`) can't be "
               "provided\n");
        REPORT_FATAL_ERROR();
    }

    if (!reflectance) {
        if (!eta) {
            auto spectrum_cu_eta =
                parameters.get_spectrum("metal-Cu-eta", SpectrumType::Albedo, gpu_dynamic_pointers);
            eta = SpectrumTexture::create_constant_texture(spectrum_cu_eta, gpu_dynamic_pointers);
        }

        if (!k) {
            auto spectrum_cu_k =
                parameters.get_spectrum("metal-Cu-k", SpectrumType::Albedo, gpu_dynamic_pointers);
            k = SpectrumTexture::create_constant_texture(spectrum_cu_k, gpu_dynamic_pointers);
        }
    }

    u_roughness = parameters.get_float_texture("uroughness", gpu_dynamic_pointers);
    if (!u_roughness) {
        auto roughness_val = parameters.get_float("roughness", 0.0);
        u_roughness =
            FloatTexture::create_constant_float_texture(roughness_val, gpu_dynamic_pointers);
    }

    v_roughness = parameters.get_float_texture("vroughness", gpu_dynamic_pointers);
    if (!v_roughness) {
        auto roughness_val = parameters.get_float("roughness", 0.0);
        v_roughness =
            FloatTexture::create_constant_float_texture(roughness_val, gpu_dynamic_pointers);
    }

    remap_roughness = parameters.get_bool("remaproughness", true);

    if (!u_roughness || !v_roughness) {
        REPORT_FATAL_ERROR();
    }
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

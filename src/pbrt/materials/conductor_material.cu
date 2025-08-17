#include <pbrt/base/bxdf.h>
#include <pbrt/base/float_texture.h>
#include <pbrt/base/spectrum_texture.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/materials/conductor_material.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/textures/texture_eval_context.h>

ConductorMaterial::ConductorMaterial(const ParameterDictionary &parameters,
                                     GPUMemoryAllocator &allocator) {
    eta = parameters.get_spectrum_texture("eta", SpectrumType::Unbounded, allocator);
    k = parameters.get_spectrum_texture("k", SpectrumType::Unbounded, allocator);
    reflectance = parameters.get_spectrum_texture("reflectance", SpectrumType::Albedo, allocator);

    if (reflectance && (eta || k)) {
        printf("ERROR: for ConductorMaterial, both `reflectance` and (`eta` and `k`) can't be "
               "provided\n");
        REPORT_FATAL_ERROR();
    }

    if (!reflectance) {
        if (!eta) {
            auto spectrum_cu_eta =
                parameters.get_spectrum("metal-Cu-eta", SpectrumType::Unbounded, allocator);
            eta = SpectrumTexture::create_constant_texture(spectrum_cu_eta, allocator);
        }

        if (!k) {
            auto spectrum_cu_k =
                parameters.get_spectrum("metal-Cu-k", SpectrumType::Unbounded, allocator);
            k = SpectrumTexture::create_constant_texture(spectrum_cu_k, allocator);
        }
    }

    uRoughness = parameters.get_float_texture_or_null("uroughness", allocator);
    if (!uRoughness) {
        uRoughness = parameters.get_float_texture("roughness", 0.0, allocator);
    }

    vRoughness = parameters.get_float_texture_or_null("vroughness", allocator);
    if (!vRoughness) {
        vRoughness = parameters.get_float_texture("roughness", 0.0, allocator);
    }

    remapRoughness = parameters.get_bool("remaproughness", true);

    if (!uRoughness || !vRoughness) {
        REPORT_FATAL_ERROR();
    }
}

PBRT_CPU_GPU
BxDF ConductorMaterial::get_bxdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const {
    auto uRough = uRoughness->evaluate(ctx);
    auto vRough = vRoughness->evaluate(ctx);

    if (remapRoughness) {
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

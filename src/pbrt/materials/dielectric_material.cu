#include <pbrt/base/float_texture.h>
#include <pbrt/base/material.h>
#include <pbrt/base/spectrum.h>
#include <pbrt/bxdfs/dielectric_bxdf.h>
#include <pbrt/materials/dielectric_material.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/textures/float_constant_texture.h>

void DielectricMaterial::init(const ParameterDictionary &parameters,
                              GPUMemoryAllocator &allocator) {
    eta = nullptr;

    auto key_eta = "eta";
    if (parameters.has_floats(key_eta)) {
        eta = Spectrum::create_constant_spectrum(parameters.get_float(key_eta), allocator);
    } else {
        eta = parameters.get_spectrum(key_eta, SpectrumType::Unbounded, allocator);
    }

    if (!eta) {
        eta = Spectrum::create_constant_spectrum(1.5, allocator);
    }

    uRoughness = parameters.get_float_texture_or_null("uroughness", allocator);
    if (!uRoughness) {
        auto roughness_val = parameters.get_float("roughness", 0.0);
        uRoughness = FloatTexture::create_constant_float_texture(roughness_val, allocator);
    }

    vRoughness = parameters.get_float_texture_or_null("vroughness", allocator);
    if (!vRoughness) {
        auto roughness_val = parameters.get_float("roughness", 0.0);
        vRoughness = FloatTexture::create_constant_float_texture(roughness_val, allocator);
    }

    remapRoughness = parameters.get_bool("remaproughness", true);

    if (eta == nullptr) {
        REPORT_FATAL_ERROR();
    }
}

PBRT_CPU_GPU
DielectricBxDF DielectricMaterial::get_dielectric_bsdf(const MaterialEvalContext &ctx,
                                                       SampledWavelengths &lambda) const {
    // Compute index of refraction for dielectric material
    Real sampled_eta = (*eta)(lambda[0]);
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

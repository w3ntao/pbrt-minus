#include "pbrt/base/float_texture.h"
#include "pbrt/base/material.h"
#include "pbrt/base/spectrum_texture.h"
#include "pbrt/bxdfs/coated_conductor_bxdf.h"
#include "pbrt/materials/coated_conductor_material.h"
#include "pbrt/scene/parameter_dictionary.h"

const FloatTexture *build_float_texture(const std::string &primary_key,
                                        const std::string &secondary_key, FloatType val,
                                        const ParameterDictionary &parameters,
                                        std::vector<void *> &gpu_dynamic_pointers) {
    auto texture = parameters.get_float_texture_or_null(primary_key, gpu_dynamic_pointers);

    if (!texture) {
        texture = parameters.get_float_texture_or_null(secondary_key, gpu_dynamic_pointers);
    }

    if (!texture) {
        texture = FloatTexture::create_constant_float_texture(val, gpu_dynamic_pointers);
    }

    return texture;
}

void CoatedConductorMaterial::init(const ParameterDictionary &parameters,
                                   std::vector<void *> &gpu_dynamic_pointers) {
    interfaceURoughness = build_float_texture("interface.uroughness", "interface.roughness", 0.0,
                                              parameters, gpu_dynamic_pointers);

    interfaceVRoughness = build_float_texture("interface.vroughness", "interface.roughness", 0.0,
                                              parameters, gpu_dynamic_pointers);

    thickness = parameters.get_float_texture("thickness", 0.01, gpu_dynamic_pointers);

    interfaceEta = nullptr;
    auto key_interface_eta = "interface.eta";
    if (parameters.has_floats(key_interface_eta)) {
        interfaceEta = Spectrum::create_constant_spectrum(
            parameters.get_float(key_interface_eta, {}), gpu_dynamic_pointers);
    } else {
        interfaceEta =
            parameters.get_spectrum("interface.eta", SpectrumType::Unbounded, gpu_dynamic_pointers);
    }

    if (!interfaceEta) {
        interfaceEta = Spectrum::create_constant_spectrum(1.5, gpu_dynamic_pointers);
    }

    conductorURoughness = build_float_texture("conductor.uroughness", "conductor.roughness", 0.0,
                                              parameters, gpu_dynamic_pointers);

    conductorVRoughness = build_float_texture("conductor.vroughness", "conductor.roughness", 0.0,
                                              parameters, gpu_dynamic_pointers);

    conductorEta = parameters.get_spectrum_texture("conductor.eta", SpectrumType::Unbounded,
                                                   gpu_dynamic_pointers);
    k = parameters.get_spectrum_texture("conductor.k", SpectrumType::Unbounded,
                                        gpu_dynamic_pointers);
    reflectance =
        parameters.get_spectrum_texture("reflectance", SpectrumType::Albedo, gpu_dynamic_pointers);

    if (reflectance && (conductorEta || k)) {
        printf("ERROR: for CoatedConductorMaterial,"
               " both `reflectance` and (`eta` and `k`) can't be provided.");
        REPORT_FATAL_ERROR();
    }

    if (!reflectance) {
        if (!conductorEta) {
            auto cu_eta = parameters.get_spectrum("metal-Cu-eta", SpectrumType::Unbounded,
                                                  gpu_dynamic_pointers);
            conductorEta = SpectrumTexture::create_constant_texture(cu_eta, gpu_dynamic_pointers);
        }

        if (!k) {
            auto cu_k = parameters.get_spectrum("metal-Cu-k", SpectrumType::Unbounded,
                                                gpu_dynamic_pointers);
            k = SpectrumTexture::create_constant_texture(cu_k, gpu_dynamic_pointers);
        }
    }

    maxDepth = parameters.get_integer("maxdepth", 10);
    nSamples = parameters.get_integer("nsamples", 1);

    g = parameters.get_float_texture("g", 0.0, gpu_dynamic_pointers);

    albedo = parameters.get_spectrum_texture("albedo", SpectrumType::Albedo, gpu_dynamic_pointers);
    if (!albedo) {
        auto spectrum = Spectrum::create_constant_spectrum(0.0, gpu_dynamic_pointers);
        albedo = SpectrumTexture::create_constant_texture(spectrum, gpu_dynamic_pointers);
    }

    remapRoughness = parameters.get_bool("remaproughness", true);
}

PBRT_GPU
CoatedConductorBxDF
CoatedConductorMaterial::get_coated_conductor_bsdf(const MaterialEvalContext &ctx,
                                                   SampledWavelengths &lambda) const {
    auto iurough = interfaceURoughness->evaluate(ctx);
    auto ivrough = interfaceVRoughness->evaluate(ctx);
    if (remapRoughness) {
        iurough = TrowbridgeReitzDistribution::RoughnessToAlpha(iurough);
        ivrough = TrowbridgeReitzDistribution::RoughnessToAlpha(ivrough);
    }
    TrowbridgeReitzDistribution interfaceDistrib(iurough, ivrough);

    auto thick = thickness->evaluate(ctx);
    auto ieta = interfaceEta->operator()(lambda[0]);

    if (!interfaceEta->is_constant_spectrum()) {
        lambda.terminate_secondary();
    }
    if (ieta == 0) {
        ieta = 1;
    }

    SampledSpectrum ce, ck;
    if (conductorEta) {
        ce = conductorEta->evaluate(ctx, lambda);
        ck = k->evaluate(ctx, lambda);
    } else {
        // Avoid r==1 NaN case...
        SampledSpectrum r = reflectance->evaluate(ctx, lambda).clamp(0, .9999);
        ce = SampledSpectrum(1.f);
        ck = 2.0 * r.sqrt() / (SampledSpectrum(1.0) - r).clamp(0.0, Infinity).sqrt();
    }
    ce /= ieta;
    ck /= ieta;

    auto curough = conductorURoughness->evaluate(ctx);
    auto cvrough = conductorVRoughness->evaluate(ctx);
    if (remapRoughness) {
        curough = TrowbridgeReitzDistribution::RoughnessToAlpha(curough);
        cvrough = TrowbridgeReitzDistribution::RoughnessToAlpha(cvrough);
    }
    TrowbridgeReitzDistribution conductorDistrib(curough, cvrough);

    SampledSpectrum a = albedo->evaluate(ctx, lambda).clamp(0, 1);
    auto gg = clamp<FloatType>(g->evaluate(ctx), -1, 1);

    return CoatedConductorBxDF(DielectricBxDF(ieta, interfaceDistrib),
                               ConductorBxDF(conductorDistrib, ce, ck), thick, a, gg, maxDepth,
                               nSamples);
}

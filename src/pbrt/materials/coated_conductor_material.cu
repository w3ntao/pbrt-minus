#include <pbrt/base/float_texture.h>
#include <pbrt/base/material.h>
#include <pbrt/base/spectrum_texture.h>
#include <pbrt/bxdfs/coated_conductor_bxdf.h>
#include <pbrt/materials/coated_conductor_material.h>
#include <pbrt/scene/parameter_dictionary.h>

const FloatTexture *build_float_texture(const std::string &primary_key,
                                        const std::string &secondary_key, FloatType val,
                                        const ParameterDictionary &parameters,
                                        GPUMemoryAllocator &allocator) {
    auto texture = parameters.get_float_texture_or_null(primary_key, allocator);

    if (!texture) {
        texture = parameters.get_float_texture_or_null(secondary_key, allocator);
    }

    if (!texture) {
        texture = FloatTexture::create_constant_float_texture(val, allocator);
    }

    return texture;
}

void CoatedConductorMaterial::init(const ParameterDictionary &parameters,
                                   GPUMemoryAllocator &allocator) {
    interfaceURoughness = build_float_texture("interface.uroughness", "interface.roughness", 0.0,
                                              parameters, allocator);

    interfaceVRoughness = build_float_texture("interface.vroughness", "interface.roughness", 0.0,
                                              parameters, allocator);

    thickness = parameters.get_float_texture("thickness", 0.01, allocator);

    interfaceEta = nullptr;
    auto key_interface_eta = "interface.eta";
    if (parameters.has_floats(key_interface_eta)) {
        interfaceEta = Spectrum::create_constant_spectrum(
            parameters.get_float(key_interface_eta, {}), allocator);
    } else {
        interfaceEta = parameters.get_spectrum("interface.eta", SpectrumType::Unbounded, allocator);
    }

    if (!interfaceEta) {
        interfaceEta = Spectrum::create_constant_spectrum(1.5, allocator);
    }

    conductorURoughness = build_float_texture("conductor.uroughness", "conductor.roughness", 0.0,
                                              parameters, allocator);

    conductorVRoughness = build_float_texture("conductor.vroughness", "conductor.roughness", 0.0,
                                              parameters, allocator);

    conductorEta =
        parameters.get_spectrum_texture("conductor.eta", SpectrumType::Unbounded, allocator);
    k = parameters.get_spectrum_texture("conductor.k", SpectrumType::Unbounded, allocator);
    reflectance = parameters.get_spectrum_texture("reflectance", SpectrumType::Albedo, allocator);

    if (reflectance && (conductorEta || k)) {
        printf("ERROR: for CoatedConductorMaterial,"
               " both `reflectance` and (`eta` and `k`) can't be provided.");
        REPORT_FATAL_ERROR();
    }

    if (!reflectance) {
        if (!conductorEta) {
            auto cu_eta =
                parameters.get_spectrum("metal-Cu-eta", SpectrumType::Unbounded, allocator);
            conductorEta = SpectrumTexture::create_constant_texture(cu_eta, allocator);
        }

        if (!k) {
            auto cu_k = parameters.get_spectrum("metal-Cu-k", SpectrumType::Unbounded, allocator);
            k = SpectrumTexture::create_constant_texture(cu_k, allocator);
        }
    }

    maxDepth = parameters.get_integer("maxdepth", 10);
    nSamples = parameters.get_integer("nsamples", 1);

    g = parameters.get_float_texture("g", 0.0, allocator);

    albedo = parameters.get_spectrum_texture("albedo", SpectrumType::Albedo, allocator);
    if (!albedo) {
        auto spectrum = Spectrum::create_constant_spectrum(0.0, allocator);
        albedo = SpectrumTexture::create_constant_texture(spectrum, allocator);
    }

    remapRoughness = parameters.get_bool("remaproughness", true);
}

PBRT_CPU_GPU
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

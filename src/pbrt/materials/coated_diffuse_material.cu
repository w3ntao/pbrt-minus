#include "pbrt/materials/coated_diffuse_material.h"

#include "pbrt/base/material.h"
#include "pbrt/base/texture.h"
#include "pbrt/base/spectrum.h"
#include "pbrt/scene/parameter_dict.h"
#include "pbrt/spectra/constant_spectrum.h"
#include "pbrt/textures/spectrum_constant_texture.h"

void CoatedDiffuseMaterial::init(const ParameterDict &parameters,
                                 std::vector<void *> &gpu_dynamic_pointers) {
    auto reflectance_key = "reflectance";
    if (parameters.has_spectrum_texture(reflectance_key)) {
        reflectance = parameters.get_spectrum_texture(reflectance_key);
    } else {
        // TODO: rewrite this part
        ConstantSpectrum *constant_spectrum;
        Spectrum *spectrum;
        SpectrumConstantTexture *spectrum_constant_texture;
        SpectrumTexture *spectrum_texture;

        CHECK_CUDA_ERROR(cudaMallocManaged(&constant_spectrum, sizeof(ConstantSpectrum)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum, sizeof(Spectrum)));
        CHECK_CUDA_ERROR(
            cudaMallocManaged(&spectrum_constant_texture, sizeof(SpectrumConstantTexture)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_texture, sizeof(SpectrumTexture)));
        constant_spectrum->init(0.5);
        spectrum->init(constant_spectrum);
        spectrum_constant_texture->init(spectrum);
        spectrum_texture->init(spectrum_constant_texture);

        reflectance = spectrum_texture;

        for (auto ptr : std::vector<void *>({
                 constant_spectrum,
                 spectrum,
                 spectrum_constant_texture,
                 spectrum_texture,
             })) {
            gpu_dynamic_pointers.push_back(ptr);
        }
    }
    
    auto uroughness_key = "uroughness";
    if (parameters.has_float_texture(uroughness_key)) {
        uRoughness = parameters.get_float_texture(uroughness_key);
    } else {
        auto uroughness_val = parameters.get_float(uroughness_key, 0.0);
        uRoughness = FloatTexture::create(uroughness_val, gpu_dynamic_pointers);
    }

    auto vroughness_key = "vroughness";
    if (parameters.has_float_texture(vroughness_key)) {
        vRoughness = parameters.get_float_texture(vroughness_key);
    } else {
        auto vroughness_val = parameters.get_float(vroughness_key, 0.0);
        vRoughness = FloatTexture::create(vroughness_val, gpu_dynamic_pointers);
    }

    // FloatTexture thickness = parameters.GetFloatTexture("thickness", .01, alloc);
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
    albedo = SpectrumTexture::create_constant_texture(albedo_val, gpu_dynamic_pointers);

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
    FloatType urough = uRoughness->evaluate(ctx);
    FloatType vrough = vRoughness->evaluate(ctx);

    if (remapRoughness) {
        urough = TrowbridgeReitzDistribution::RoughnessToAlpha(urough);
        vrough = TrowbridgeReitzDistribution::RoughnessToAlpha(vrough);
    }

    TrowbridgeReitzDistribution distrib(urough, vrough);

    FloatType thick = thickness->evaluate(ctx);

    FloatType sampledEta = (*eta)(lambda[0]);

    if (!eta->is_constant_spectrum()) {
        lambda.TerminateSecondary();
    }
    if (sampledEta == 0) {
        sampledEta = 1;
    }

    SampledSpectrum a = albedo->evaluate(ctx, lambda).clamp(0, 1);
    FloatType gg = clamp<FloatType>(g->evaluate(ctx), -1, 1);

    return CoatedDiffuseBxDF(DielectricBxDF(sampledEta, distrib), DiffuseBxDF(r), thick, a, gg,
                             maxDepth, nSamples);
}

#include "pbrt/materials/dielectric_material.h"

#include "pbrt/base/material.h"
#include "pbrt/base/spectrum.h"
#include "pbrt/base/texture.h"

#include "pbrt/bxdfs/dielectric_bxdf.h"
#include "pbrt/spectra/constant_spectrum.h"
#include "pbrt/scene/parameter_dict.h"
#include "pbrt/textures/float_constant_texture.h"

void DielectricMaterial::init(const ParameterDict &parameters,
                              std::vector<void *> &gpu_dynamic_pointers) {
    if (parameters.has_floats("eta")) {
        // this part is not implemented
        REPORT_FATAL_ERROR();

    } else {
        // TODO: if eta was initialized as a Spectrum?
        ConstantSpectrum *constant_eta;
        Spectrum *spectrum_eta;
        CHECK_CUDA_ERROR(cudaMallocManaged(&constant_eta, sizeof(ConstantSpectrum)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_eta, sizeof(Spectrum)));

        constant_eta->init(1.5);
        spectrum_eta->init(constant_eta);

        gpu_dynamic_pointers.push_back(constant_eta);
        gpu_dynamic_pointers.push_back(spectrum_eta);

        eta = spectrum_eta;
    }

    auto float_u_roughness = parameters.get_float("uroughness", 0.0);
    auto float_v_roughness = parameters.get_float("uroughness", 0.0);

    FloatTexture *_u_roughness;
    FloatTexture *_v_roughness;
    FloatConstantTexture *constant_u_roughness;
    FloatConstantTexture *constant_v_roughness;
    CHECK_CUDA_ERROR(cudaMallocManaged(&_u_roughness, sizeof(FloatTexture)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&_v_roughness, sizeof(FloatTexture)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&constant_u_roughness, sizeof(FloatConstantTexture)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&constant_v_roughness, sizeof(FloatConstantTexture)));

    constant_u_roughness->init(float_u_roughness);
    _u_roughness->init(constant_u_roughness);

    constant_v_roughness->init(float_v_roughness);
    _v_roughness->init(constant_v_roughness);

    u_roughness = _u_roughness;
    v_roughness = _v_roughness;

    remap_roughness = parameters.get_bool("remaproughness", true);

    for (auto ptr : std::vector<void *>{
             _u_roughness,
             _v_roughness,
             constant_u_roughness,
             constant_v_roughness,
         }) {
        gpu_dynamic_pointers.push_back(ptr);
    }
}

PBRT_GPU
DielectricBxDF DielectricMaterial::get_dielectric_bsdf(const MaterialEvalContext &ctx,
                                                       SampledWavelengths &lambda) const {
    // Compute index of refraction for dielectric material
    FloatType sampled_eta = (*eta)(lambda[0]);
    if (!eta->is_constant_spectrum()) {
        lambda.TerminateSecondary();
    }

    // Handle edge case in case lambda[0] is beyond the wavelengths stored by the
    // Spectrum.
    if (sampled_eta == 0) {
        sampled_eta = 1;
    }

    // Create microfacet distribution for dielectric material

    auto urough = u_roughness->evaluate(ctx);
    auto vrough = v_roughness->evaluate(ctx);

    if (remap_roughness) {
        urough = TrowbridgeReitzDistribution::RoughnessToAlpha(urough);
        vrough = TrowbridgeReitzDistribution::RoughnessToAlpha(vrough);
    }
    TrowbridgeReitzDistribution distrib(urough, vrough);
    return DielectricBxDF(sampled_eta, distrib);
}

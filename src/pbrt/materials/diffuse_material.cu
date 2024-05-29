#include "pbrt/materials/diffuse_material.h"

#include "pbrt/base/material.h"
#include "pbrt/base/spectrum.h"
#include "pbrt/base/texture.h"

#include "pbrt/bxdfs/diffuse_bxdf.h"
#include "pbrt/scene/parameter_dict.h"
#include "pbrt/spectra/rgb_albedo_spectrum.h"
#include "pbrt/spectrum_util/sampled_wavelengths.h"
#include "pbrt/spectrum_util/rgb_color_space.h"
#include "pbrt/textures/spectrum_constant_texture.h"

const DiffuseMaterial *DiffuseMaterial::create(const SpectrumTexture *_reflectance,
                                               std::vector<void *> &gpu_dynamic_pointers) {
    DiffuseMaterial *diffuse_material;
    CHECK_CUDA_ERROR(cudaMallocManaged(&diffuse_material, sizeof(DiffuseMaterial)));
    diffuse_material->reflectance = _reflectance;

    gpu_dynamic_pointers.push_back(diffuse_material);
    return diffuse_material;
}

void DiffuseMaterial::init(const ParameterDict &parameters,
                           std::vector<void *> &gpu_dynamic_pointers,
                           const RGBColorSpace *color_space) {
    auto key = "reflectance";

    if (parameters.has_rgb(key)) {
        SpectrumConstantTexture *spectrum_constant_texture;
        SpectrumTexture *spectrum_texture;
        CHECK_CUDA_ERROR(
            cudaMallocManaged(&spectrum_constant_texture, sizeof(SpectrumConstantTexture)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_texture, sizeof(SpectrumTexture)));

        auto rgb_val = parameters.get_rgb(key, std::nullopt);
        spectrum_constant_texture->init(
            Spectrum::create_rgb_albedo_spectrum(rgb_val, gpu_dynamic_pointers, color_space));
        spectrum_texture->init(spectrum_constant_texture);

        reflectance = spectrum_texture;

        for (auto ptr : std::vector<void *>({
                 spectrum_constant_texture,
                 spectrum_texture,
             })) {
            gpu_dynamic_pointers.push_back(ptr);
        }

        return;
    }

    if (parameters.has_spectrum_texture(key)) {
        reflectance = parameters.get_spectrum_texture(key);
        return;
    }

    REPORT_FATAL_ERROR();
}

PBRT_GPU
DiffuseBxDF DiffuseMaterial::get_diffuse_bsdf(const MaterialEvalContext &ctx,
                                              SampledWavelengths &lambda) const {
    auto r = reflectance->evaluate(ctx, lambda).clamp(0.0, 1.0);
    return DiffuseBxDF(r);
}

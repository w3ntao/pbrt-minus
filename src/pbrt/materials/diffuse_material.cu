#include "pbrt/base/material.h"
#include "pbrt/base/spectrum.h"
#include "pbrt/base/spectrum_texture.h"
#include "pbrt/bxdfs/diffuse_bxdf.h"
#include "pbrt/materials/diffuse_material.h"
#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/spectra/rgb_albedo_spectrum.h"
#include "pbrt/spectrum_util/global_spectra.h"
#include "pbrt/spectrum_util/sampled_wavelengths.h"
#include "pbrt/textures/spectrum_constant_texture.h"

const DiffuseMaterial *DiffuseMaterial::create(const SpectrumTexture *_reflectance,
                                               std::vector<void *> &gpu_dynamic_pointers) {
    DiffuseMaterial *diffuse_material;
    CHECK_CUDA_ERROR(cudaMallocManaged(&diffuse_material, sizeof(DiffuseMaterial)));
    gpu_dynamic_pointers.push_back(diffuse_material);

    diffuse_material->reflectance = _reflectance;
    return diffuse_material;
}

void DiffuseMaterial::init(const ParameterDictionary &parameters,
                           std::vector<void *> &gpu_dynamic_pointers) {
    reflectance =
        parameters.get_spectrum_texture("reflectance", SpectrumType::Albedo, gpu_dynamic_pointers);
    if (!reflectance) {
        reflectance = SpectrumTexture::create_constant_float_val_texture(0.5, gpu_dynamic_pointers);
    }
}

PBRT_CPU_GPU
DiffuseBxDF DiffuseMaterial::get_diffuse_bsdf(const MaterialEvalContext &ctx,
                                              SampledWavelengths &lambda) const {
    auto r = reflectance->evaluate(ctx, lambda).clamp(0.0, 1.0);
    return DiffuseBxDF(r);
}

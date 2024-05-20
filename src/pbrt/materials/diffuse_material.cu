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

void DiffuseMaterial::init(const RGBColorSpace *color_space, const ParameterDict &parameters,
                           std::vector<void *> &gpu_dynamic_pointers) {
    RGBAlbedoSpectrum *_reflectance_spectrum;
    Spectrum *reflectance_spectrum;
    SpectrumConstantTexture *spectrum_constant_texture;
    SpectrumTexture *spectrum_texture;

    CHECK_CUDA_ERROR(cudaMallocManaged(&_reflectance_spectrum, sizeof(RGBAlbedoSpectrum)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&reflectance_spectrum, sizeof(Spectrum)));
    CHECK_CUDA_ERROR(
        cudaMallocManaged(&spectrum_constant_texture, sizeof(SpectrumConstantTexture)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum_texture, sizeof(SpectrumTexture)));

    auto key = "reflectance";
    auto rgb_reflectance = parameters.get_rgb(key, std::nullopt);
    _reflectance_spectrum->init(color_space, rgb_reflectance);
    reflectance_spectrum->init(_reflectance_spectrum);
    spectrum_constant_texture->init(reflectance_spectrum);
    spectrum_texture->init(spectrum_constant_texture);

    reflectance = spectrum_texture;

    for (auto ptr : std::vector<void *>({
             _reflectance_spectrum,
             reflectance_spectrum,
             spectrum_constant_texture,
             spectrum_texture,
         })) {
        gpu_dynamic_pointers.push_back(ptr);
    }
}

void DiffuseMaterial::init(const SpectrumTexture *_reflectance) {
    // TODO: delete me
    reflectance = _reflectance;
}

PBRT_GPU
DiffuseBxDF DiffuseMaterial::get_diffuse_bsdf(const MaterialEvalContext &ctx,
                                              SampledWavelengths &lambda) const {
    auto r = reflectance->evaluate(ctx, lambda).clamp(0.0, 1.0);
    return DiffuseBxDF(r);
}

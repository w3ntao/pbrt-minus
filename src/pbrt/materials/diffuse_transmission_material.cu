#include <pbrt/base/spectrum_texture.h>
#include <pbrt/bxdfs/diffuse_transmission_bxdf.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/materials/diffuse_transmission_material.h>
#include <pbrt/scene/parameter_dictionary.h>

DiffuseTransmissionMaterial::DiffuseTransmissionMaterial(const ParameterDictionary &parameters,
                                                         GPUMemoryAllocator &allocator) {
    reflectance = parameters.get_spectrum_texture("reflectance", SpectrumType::Albedo, allocator);
    if (reflectance == nullptr) {
        reflectance = SpectrumTexture::create_constant_float_val_texture(0.25, allocator);
    }

    transmittance =
        parameters.get_spectrum_texture("transmittance", SpectrumType::Albedo, allocator);
    if (transmittance == nullptr) {
        transmittance = SpectrumTexture::create_constant_float_val_texture(0.25, allocator);
    }

    scale = parameters.get_float("scale", 1.0);
}

PBRT_CPU_GPU
DiffuseTransmissionBxDF
DiffuseTransmissionMaterial::get_diffuse_transmission_bsdf(const MaterialEvalContext &ctx,
                                                           SampledWavelengths &lambda) const {
    const auto r = (scale * reflectance->evaluate(ctx, lambda)).clamp(0, 1);
    const auto t = (scale * transmittance->evaluate(ctx, lambda)).clamp(0, 1);

    return DiffuseTransmissionBxDF(r, t);
}

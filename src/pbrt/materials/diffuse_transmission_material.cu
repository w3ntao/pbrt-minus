#include <pbrt/base/bxdf.h>
#include <pbrt/base/spectrum_texture.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/materials/diffuse_transmission_material.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/textures/texture_eval_context.h>

const DiffuseTransmissionMaterial *
DiffuseTransmissionMaterial::create(const ParameterDictionary &parameters,
                                    GPUMemoryAllocator &allocator) {
    auto diffuse_transmission_material = allocator.allocate<DiffuseTransmissionMaterial>();
    *diffuse_transmission_material = DiffuseTransmissionMaterial(parameters, allocator);

    return diffuse_transmission_material;
}

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
BxDF DiffuseTransmissionMaterial::get_bxdf(const MaterialEvalContext &ctx,
                                           SampledWavelengths &lambda) const {
    const auto r = (scale * reflectance->evaluate(ctx, lambda)).clamp(0, 1);
    const auto t = (scale * transmittance->evaluate(ctx, lambda)).clamp(0, 1);

    return DiffuseTransmissionBxDF(r, t);
}

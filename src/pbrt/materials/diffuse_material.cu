#include <pbrt/base/spectrum.h>
#include <pbrt/base/spectrum_texture.h>
#include <pbrt/bxdfs/diffuse_bxdf.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/materials/diffuse_material.h>
#include <pbrt/scene/parameter_dictionary.h>

DiffuseMaterial::DiffuseMaterial(const SpectrumTexture *_reflectance) : reflectance(_reflectance) {}

DiffuseMaterial::DiffuseMaterial(const ParameterDictionary &parameters,
                                 GPUMemoryAllocator &allocator) {
    reflectance = parameters.get_spectrum_texture("reflectance", SpectrumType::Albedo, allocator);
    if (!reflectance) {
        reflectance = SpectrumTexture::create_constant_float_val_texture(0.5, allocator);
    }
}

PBRT_CPU_GPU
DiffuseBxDF DiffuseMaterial::get_diffuse_bsdf(const MaterialEvalContext &ctx,
                                              SampledWavelengths &lambda) const {
    const auto r = reflectance->evaluate(ctx, lambda).clamp(0.0, 1.0);

    return DiffuseBxDF(r);
}

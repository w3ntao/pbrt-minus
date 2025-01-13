#include <pbrt/base/material.h>
#include <pbrt/base/spectrum.h>
#include <pbrt/base/spectrum_texture.h>
#include <pbrt/bxdfs/diffuse_bxdf.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/materials/diffuse_material.h>
#include <pbrt/scene/parameter_dictionary.h>

const DiffuseMaterial *DiffuseMaterial::create(const SpectrumTexture *_reflectance,
                                               GPUMemoryAllocator &allocator) {
    auto diffuse_material = allocator.allocate<DiffuseMaterial>();

    diffuse_material->reflectance = _reflectance;
    return diffuse_material;
}

void DiffuseMaterial::init(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator) {
    reflectance = parameters.get_spectrum_texture("reflectance", SpectrumType::Albedo, allocator);
    if (!reflectance) {
        reflectance = SpectrumTexture::create_constant_float_val_texture(0.5, allocator);
    }
}

PBRT_CPU_GPU
DiffuseBxDF DiffuseMaterial::get_diffuse_bsdf(const MaterialEvalContext &ctx,
                                              SampledWavelengths &lambda) const {
    auto r = reflectance->evaluate(ctx, lambda).clamp(0.0, 1.0);
    return DiffuseBxDF(r);
}

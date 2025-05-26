#include <pbrt/base/bxdf.h>
#include <pbrt/base/spectrum_texture.h>
#include <pbrt/bxdfs/diffuse_bxdf.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/materials/diffuse_material.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/textures/texture_eval_context.h>

const DiffuseMaterial *DiffuseMaterial::create(const ParameterDictionary &parameters,
                                               GPUMemoryAllocator &allocator) {
    auto diffuse_material = allocator.allocate<DiffuseMaterial>();
    *diffuse_material = DiffuseMaterial(parameters, allocator);

    return diffuse_material;
}

const DiffuseMaterial *DiffuseMaterial::create(const SpectrumTexture *reflectance,
                                               GPUMemoryAllocator &allocator) {
    auto diffuse_material = allocator.allocate<DiffuseMaterial>();
    *diffuse_material = DiffuseMaterial(reflectance);

    return diffuse_material;
}

DiffuseMaterial::DiffuseMaterial(const SpectrumTexture *_reflectance) : reflectance(_reflectance) {}

DiffuseMaterial::DiffuseMaterial(const ParameterDictionary &parameters,
                                 GPUMemoryAllocator &allocator) {
    reflectance = parameters.get_spectrum_texture("reflectance", SpectrumType::Albedo, allocator);
    if (!reflectance) {
        reflectance = SpectrumTexture::create_constant_float_val_texture(0.5, allocator);
    }
}

PBRT_CPU_GPU
BxDF DiffuseMaterial::get_bxdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const {
    const auto r = reflectance->evaluate(ctx, lambda).clamp(0.0, 1.0);

    return DiffuseBxDF(r);
}

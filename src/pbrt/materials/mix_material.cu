#include <pbrt/base/bxdf.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/materials/mix_material.h>
#include <pbrt/scene/parameter_dictionary.h>

const MixMaterial *MixMaterial::create(const ParameterDictionary &parameters,
                                       GPUMemoryAllocator &allocator) {
    auto mix_material = allocator.allocate<MixMaterial>();
    *mix_material = MixMaterial(parameters);

    return mix_material;
}

MixMaterial::MixMaterial(const ParameterDictionary &parameters) {
    amount = parameters.get_float("amount", 0.5);

    if (amount <= 0.0 || amount >= 1.0) {
        REPORT_FATAL_ERROR();
    }

    const auto str_materials = parameters.get_strings("materials");
    materials[0] = parameters.get_material(str_materials[0]);
    materials[1] = parameters.get_material(str_materials[1]);
}

PBRT_CPU_GPU
const Material *MixMaterial::get_material(const Real u) const {
    return u < amount ? materials[0] : materials[1];
}

PBRT_CPU_GPU
BxDF MixMaterial::get_bxdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const {
    REPORT_FATAL_ERROR();
    return BxDF();
}

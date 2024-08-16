#include "pbrt/materials/mix_material.h"

#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/util/hash.h"

void MixMaterial::init(const ParameterDictionary &parameters) {
    amount = parameters.get_float("amount", 0.5);

    if (amount <= 0.0 || amount >= 1.0) {
        REPORT_FATAL_ERROR();
    }

    auto str_materials = parameters.get_strings("materials");
    materials[0] = parameters.get_material(str_materials[0]);
    materials[1] = parameters.get_material(str_materials[1]);
}

PBRT_CPU_GPU
const Material *MixMaterial::get_material(const SurfaceInteraction *si) const {
    auto u = pstd::hash_float(si->pi, si->wo, materials[0], materials[1]);
    // TODO: test hash_float range
    return u < amount ? materials[0] : materials[1];
}

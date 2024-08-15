#include "pbrt/materials/coated_conductor_material.h"

#include "pbrt/base/float_texture.h"
#include "pbrt/scene/parameter_dictionary.h"

const FloatTexture *build_float_texture(const std::string &primary_key,
                                        const std::string &secondary_key, FloatType val,
                                        const ParameterDictionary &parameters,
                                        std::vector<void *> &gpu_dynamic_pointers) {
    auto texture = parameters.get_float_texture(primary_key, gpu_dynamic_pointers);
    if (!texture) {
        texture = parameters.get_float_texture(secondary_key, gpu_dynamic_pointers);
    }
    if (!texture) {
        texture = FloatTexture::create_constant_float_texture(val, gpu_dynamic_pointers);
    }
    return texture;
}

void CoatedConductorMaterial::init(const ParameterDictionary &parameters,
                                   std::vector<void *> &gpu_dynamic_pointers) {
    interfaceURoughness = build_float_texture("interface.uroughness", "interface.roughness", 0.0,
                                              parameters, gpu_dynamic_pointers);

    interfaceVRoughness = build_float_texture("interface.vroughness", "interface.roughness", 0.0,
                                              parameters, gpu_dynamic_pointers);

    thickness = parameters.get_float_texture("thickness", gpu_dynamic_pointers);
    if (!thickness) {
        thickness = FloatTexture::create_constant_float_texture(0.01, gpu_dynamic_pointers);
    }

    interfaceEta = nullptr;
    auto key_interface_eta = "interface.eta";
    if (parameters.has_floats(key_interface_eta)) {
        interfaceEta = Spectrum::create_constant_spectrum(
            parameters.get_float(key_interface_eta, {}), gpu_dynamic_pointers);
    } else {
        interfaceEta =
            parameters.get_spectrum("interface.eta", SpectrumType::Unbounded, gpu_dynamic_pointers);
    }

    if (!interfaceEta) {
        interfaceEta = Spectrum::create_constant_spectrum(1.5, gpu_dynamic_pointers);
    }

    conductorURoughness = build_float_texture("conductor.uroughness", "conductor.roughness", 0.0,
                                              parameters, gpu_dynamic_pointers);

    conductorVRoughness = build_float_texture("conductor.vroughness", "conductor.roughness", 0.0,
                                              parameters, gpu_dynamic_pointers);

    /*
     SpectrumTexture conductorEta = parameters.GetSpectrumTextureOrNull(
"conductor.eta", SpectrumType::Unbounded, alloc);
SpectrumTexture k = parameters.GetSpectrumTextureOrNull(
"conductor.k", SpectrumType::Unbounded, alloc);
SpectrumTexture reflectance =
parameters.GetSpectrumTextureOrNull("reflectance", SpectrumType::Albedo, alloc);
     */

    // TODO: progress 2024/08/14: implementing CoatedConductorMaterial
    REPORT_FATAL_ERROR();
}

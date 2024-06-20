#include "pbrt/integrators/ambient_occlusion.h"

#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/spectrum_util/global_spectra.h"

const AmbientOcclusionIntegrator *
AmbientOcclusionIntegrator::create(const ParameterDictionary &parameters,
                                   const IntegratorBase *integrator_base,
                                   std::vector<void *> &gpu_dynamic_pointers) {

    auto illuminant_spectrum = parameters.global_spectra->rgb_color_space->illuminant;

    const auto cie_y = parameters.global_spectra->cie_y;
    auto illuminant_scale = 1.0 / illuminant_spectrum->to_photometric(cie_y);

    AmbientOcclusionIntegrator *ambient_occlusion_integrator;
    CHECK_CUDA_ERROR(
        cudaMallocManaged(&ambient_occlusion_integrator, sizeof(AmbientOcclusionIntegrator)));
    gpu_dynamic_pointers.push_back(ambient_occlusion_integrator);

    ambient_occlusion_integrator->init(integrator_base, illuminant_spectrum, illuminant_scale);

    return ambient_occlusion_integrator;
}

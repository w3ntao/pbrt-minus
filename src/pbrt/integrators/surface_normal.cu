#include "pbrt/integrators/surface_normal.h"

#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/spectrum_util/global_spectra.h"

const SurfaceNormalIntegrator *
SurfaceNormalIntegrator::create(const ParameterDictionary &parameters,
                                const IntegratorBase *integrator_base,
                                std::vector<void *> &gpu_dynamic_pointers) {
    SurfaceNormalIntegrator *surface_normal_integrator;
    CHECK_CUDA_ERROR(
        cudaMallocManaged(&surface_normal_integrator, sizeof(SurfaceNormalIntegrator)));
    gpu_dynamic_pointers.push_back(surface_normal_integrator);

    surface_normal_integrator->init(integrator_base, parameters.global_spectra->rgb_color_space);
    return surface_normal_integrator;
}
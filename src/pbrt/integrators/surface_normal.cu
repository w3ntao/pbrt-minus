#include "pbrt/base/interaction.h"
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

PBRT_CPU_GPU
SampledSpectrum SurfaceNormalIntegrator::li(const Ray &ray, SampledWavelengths &lambda) const {
    const auto shape_intersection = base->intersect(ray, Infinity);
    if (!shape_intersection) {
        return SampledSpectrum(0.0);
    }

    const Vector3f normal =
        shape_intersection->interaction.shading.n.to_vector3().face_forward(-ray.d).normalize();

    const auto color = normal.softmax();

    return color[0] * rgb_spectra[0].sample(lambda) + color[1] * rgb_spectra[1].sample(lambda) +
           color[2] * rgb_spectra[2].sample(lambda);
}

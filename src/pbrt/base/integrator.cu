#include "pbrt/base/integrator.h"
#include "pbrt/base/ray.h"
#include "pbrt/accelerator/hlbvh.h"
#include "pbrt/spectra/sampled_wavelengths.h"

#include "pbrt/integrators/surface_normal.h"
#include "pbrt/integrators/ambient_occlusion.h"

void Integrator::init(const AmbientOcclusionIntegrator *ambient_occlusion_integrator) {
    integrator_type = IntegratorType::ambient_occlusion;
    integrator_ptr = ambient_occlusion_integrator;
}

void Integrator::init(const SurfaceNormalIntegrator *surface_normal_integrator) {
    integrator_type = IntegratorType::surface_normal;
    integrator_ptr = surface_normal_integrator;
}

PBRT_GPU
SampledSpectrum Integrator::li(const Ray &ray, SampledWavelengths &lambda, const HLBVH *bvh,
                               Sampler &sampler) const {
    switch (integrator_type) {
    case (IntegratorType::surface_normal): {
        return ((SurfaceNormalIntegrator *)integrator_ptr)->li(ray, lambda, bvh, sampler);
    }

    case (IntegratorType::ambient_occlusion): {
        return ((AmbientOcclusionIntegrator *)integrator_ptr)->li(ray, lambda, bvh, sampler);
    }
    }

    report_error();
}

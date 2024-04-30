#include "pbrt/base/integrator.h"

#include "pbrt/base/sampler.h"
#include "pbrt/base/ray.h"
#include "pbrt/accelerator/hlbvh.h"
#include "pbrt/spectrum_util/sampled_wavelengths.h"

#include "pbrt/integrators/surface_normal.h"
#include "pbrt/integrators/ambient_occlusion.h"
#include "pbrt/integrators/random_walk.h"

void Integrator::init(const AmbientOcclusionIntegrator *ambient_occlusion_integrator) {
    integrator_type = IntegratorType::ambient_occlusion;
    integrator_ptr = ambient_occlusion_integrator;
}

void Integrator::init(const SurfaceNormalIntegrator *surface_normal_integrator) {
    integrator_type = IntegratorType::surface_normal;
    integrator_ptr = surface_normal_integrator;
}

void Integrator::init(const RandomWalkIntegrator *random_walk_integrator) {
    integrator_type = IntegratorType::random_walk;
    integrator_ptr = random_walk_integrator;
}

PBRT_GPU
SampledSpectrum Integrator::li(const Ray &ray, SampledWavelengths &lambda, const HLBVH *bvh,
                               Sampler *sampler) const {
    switch (integrator_type) {
    case (IntegratorType::surface_normal): {
        return ((SurfaceNormalIntegrator *)integrator_ptr)->li(ray, lambda, bvh, sampler);
    }

    case (IntegratorType::ambient_occlusion): {
        return ((AmbientOcclusionIntegrator *)integrator_ptr)->li(ray, lambda, bvh, sampler);
    }

    case (IntegratorType::random_walk): {
        return ((RandomWalkIntegrator *)integrator_ptr)->li(ray, lambda, bvh, sampler);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

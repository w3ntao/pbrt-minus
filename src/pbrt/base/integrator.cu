#include "pbrt/base/integrator.h"

#include "pbrt/base/sampler.h"
#include "pbrt/base/ray.h"
#include "pbrt/spectrum_util/sampled_wavelengths.h"

#include "pbrt/integrators/surface_normal.h"
#include "pbrt/integrators/ambient_occlusion.h"
#include "pbrt/integrators/random_walk.h"
#include "pbrt/integrators/simple_path.h"

void Integrator::init(const AmbientOcclusionIntegrator *ambient_occlusion_integrator) {
    integrator_type = Type::ambient_occlusion;
    integrator_ptr = ambient_occlusion_integrator;
}

void Integrator::init(const SurfaceNormalIntegrator *surface_normal_integrator) {
    integrator_type = Type::surface_normal;
    integrator_ptr = surface_normal_integrator;
}

void Integrator::init(const RandomWalkIntegrator *random_walk_integrator) {
    integrator_type = Type::random_walk;
    integrator_ptr = random_walk_integrator;
}

void Integrator::init(const SimplePathIntegrator *simple_path_integrator) {
    integrator_type = Type::simple_path;
    integrator_ptr = simple_path_integrator;
}

PBRT_GPU
SampledSpectrum Integrator::li(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler) const {
    switch (integrator_type) {
    case (Type::surface_normal): {
        return ((SurfaceNormalIntegrator *)integrator_ptr)->li(ray, lambda);
    }

    case (Type::ambient_occlusion): {
        return ((AmbientOcclusionIntegrator *)integrator_ptr)->li(ray, lambda, sampler);
    }

    case (Type::random_walk): {
        return ((RandomWalkIntegrator *)integrator_ptr)->li(ray, lambda, sampler);
    }

    case (Type::simple_path): {
        return ((SimplePathIntegrator *)integrator_ptr)->li(ray, lambda, sampler);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

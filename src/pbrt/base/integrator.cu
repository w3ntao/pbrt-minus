#include "pbrt/base/integrator.h"

#include "pbrt/base/sampler.h"
#include "pbrt/base/ray.h"
#include "pbrt/spectrum_util/sampled_wavelengths.h"

#include "pbrt/integrators/surface_normal.h"
#include "pbrt/integrators/ambient_occlusion.h"
#include "pbrt/integrators/random_walk.h"
#include "pbrt/integrators/simple_path.h"

void Integrator::init(const AmbientOcclusionIntegrator *ambient_occlusion_integrator) {
    type = Type::ambient_occlusion;
    ptr = ambient_occlusion_integrator;
}

void Integrator::init(const SurfaceNormalIntegrator *surface_normal_integrator) {
    type = Type::surface_normal;
    ptr = surface_normal_integrator;
}

void Integrator::init(const RandomWalkIntegrator *random_walk_integrator) {
    type = Type::random_walk;
    ptr = random_walk_integrator;
}

void Integrator::init(const SimplePathIntegrator *simple_path_integrator) {
    type = Type::simple_path;
    ptr = simple_path_integrator;
}

PBRT_GPU
SampledSpectrum Integrator::li(const DifferentialRay &ray, SampledWavelengths &lambda,
                               Sampler *sampler) const {
    switch (type) {
    case (Type::surface_normal): {
        return ((SurfaceNormalIntegrator *)ptr)->li(ray, lambda);
    }

    case (Type::ambient_occlusion): {
        return ((AmbientOcclusionIntegrator *)ptr)->li(ray, lambda, sampler);
    }

    case (Type::random_walk): {
        return ((RandomWalkIntegrator *)ptr)->li(ray, lambda, sampler);
    }

    case (Type::simple_path): {
        return ((SimplePathIntegrator *)ptr)->li(ray, lambda, sampler);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

#pragma once

#include "pbrt/integrators/integrator_base.h"
#include "pbrt/light_samplers/uniform_light_sampler.h"

class Sampler;

class SimplePathIntegrator {
  public:
    void init(const IntegratorBase *_base, uint _max_depth);

    PBRT_GPU
    SampledSpectrum li(const DifferentialRay &ray, SampledWavelengths &lambda, Sampler *sampler);

    PBRT_GPU
    bool fast_intersect(const Ray &ray, FloatType t_max) const;

    PBRT_GPU
    bool unoccluded(const Interaction &p0, const Interaction &p1) const {
        return !fast_intersect(p0.spawn_ray_to(p1), 0.6) &&
               !fast_intersect(p1.spawn_ray_to(p0), 0.6);
    }

  private:
    const IntegratorBase *base;
    uint max_depth;
};

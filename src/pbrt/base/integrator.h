#pragma once

#include "pbrt/spectrum_util/sampled_spectrum.h"

class Ray;
class SampledWavelengths;
class HLBVH;
class AmbientOcclusionIntegrator;
class SurfaceNormalIntegrator;
class RandomWalkIntegrator;
class Sampler;

class Integrator {
  public:
    void init(const AmbientOcclusionIntegrator *ambient_occlusion_integrator);

    void init(const SurfaceNormalIntegrator *surface_normal_integrator);

    void init(const RandomWalkIntegrator *random_walk_integrator);

    PBRT_GPU SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda, const HLBVH *bvh,
                                Sampler *sampler) const;

  private:
    enum class IntegratorType {
        surface_normal,
        ambient_occlusion,
        random_walk,
    };

    IntegratorType integrator_type;
    const void *integrator_ptr;
};

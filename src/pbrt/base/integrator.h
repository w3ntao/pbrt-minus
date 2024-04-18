#pragma once

#include "pbrt/base/sampler.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"

class Ray;
class SampledWavelengths;
class HLBVH;
class AmbientOcclusionIntegrator;
class SurfaceNormalIntegrator;
class RandomWalkIntegrator;

class Integrator {
  public:
    void init(const AmbientOcclusionIntegrator *ambient_occlusion_integrator);

    void init(const SurfaceNormalIntegrator *surface_normal_integrator);

    void init(const RandomWalkIntegrator *random_walk_integrator);

    PBRT_GPU SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda, const HLBVH *bvh,
                                Sampler &sampler) const;

  private:
    enum class IntegratorType {
        surface_normal,
        ambient_occlusion,
        random_walk,
    };

    IntegratorType integrator_type;
    const void *integrator_ptr;

    PBRT_CPU_GPU void report_error() const {
        printf("\nIntegrator: this type is not implemented\n");
#if defined(__CUDA_ARCH__)
        asm("trap;");
#else
        throw std::runtime_error("Integrator: this type is not implemented\n");
#endif
    }
};

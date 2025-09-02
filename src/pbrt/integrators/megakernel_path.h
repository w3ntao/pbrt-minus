#pragma once

#include <pbrt/base/ray.h>
#include <pbrt/scene/parameter_dictionary.h>

class Sampler;
struct IntegratorBase;

class MegakernelPathIntegrator {
  public:
    MegakernelPathIntegrator(const ParameterDictionary &parameters,
                             const IntegratorBase *integrator_base)
        : base(integrator_base) {
        max_depth = parameters.get_integer("maxdepth", 5);
        regularize = parameters.get_bool("regularize", false);
    }

    PBRT_CPU_GPU
    static SampledSpectrum evaluate_Li_volume(const Ray &primary_ray, SampledWavelengths &lambda,
                                              Sampler *sampler, const IntegratorBase *base,
                                              int max_depth, bool regularize);

    PBRT_CPU_GPU
    static SampledSpectrum sample_Ld_volume(const SurfaceInteraction &surface_interaction,
                                            const BSDF *bsdf, const SampledWavelengths &lambda,
                                            Sampler *sampler, const IntegratorBase *base,
                                            int max_depth);

    PBRT_CPU_GPU
    SampledSpectrum li(const Ray &primary_ray, SampledWavelengths &lambda, Sampler *sampler) const;

  private:
    const IntegratorBase *base = nullptr;
    int max_depth = 0;
    bool regularize = true;
};

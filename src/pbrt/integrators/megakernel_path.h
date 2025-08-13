#pragma once

#include <pbrt/gpu/macro.h>

class BSDF;
class GPUMemoryAllocator;
class Sampler;
class ParameterDictionary;
struct IntegratorBase;

class MegakernelPathIntegrator {
  public:
    static const MegakernelPathIntegrator *create(const ParameterDictionary &parameters,
                                                  const IntegratorBase *integrator_base,
                                                  GPUMemoryAllocator &allocator);

    MegakernelPathIntegrator(const IntegratorBase *_base, int _max_depth, bool _regularize)
        : base(_base), max_depth(_max_depth), regularize(_regularize) {}

    PBRT_CPU_GPU
    static SampledSpectrum evaluate_li_volume(const Ray &primary_ray, SampledWavelengths &lambda,
                                              const IntegratorBase *base, Sampler *sampler,
                                              int max_depth, bool regularize);

    PBRT_CPU_GPU
    static inline SampledSpectrum sample_ld_volume(const SurfaceInteraction &surface_interaction,
                                                   const BSDF *bsdf, SampledWavelengths &lambda,
                                                   const IntegratorBase *base, Sampler *sampler,
                                                   int max_depth);

    PBRT_CPU_GPU
    SampledSpectrum li(const Ray &primary_ray, SampledWavelengths &lambda, Sampler *sampler) const;

  private:
    const IntegratorBase *base = nullptr;
    int max_depth = 5; // TODO: move max_depth into IntegratorBase
    bool regularize = true;
};

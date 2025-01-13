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

    void init(const IntegratorBase *_base, uint _max_depth, bool _regularize);

    PBRT_CPU_GPU
    static SampledSpectrum evaluate_li(const Ray &primary_ray, SampledWavelengths &lambda,
                                       const IntegratorBase *base, Sampler *sampler, uint max_depth,
                                       bool regularize);

    PBRT_CPU_GPU
    static inline SampledSpectrum sample_ld(const SurfaceInteraction &intr, const BSDF *bsdf,
                                            SampledWavelengths &lambda, const IntegratorBase *base,
                                            Sampler *sampler);

    PBRT_CPU_GPU
    SampledSpectrum li(const Ray &primary_ray, SampledWavelengths &lambda, Sampler *sampler) const;

  private:
    const IntegratorBase *base;
    uint max_depth;
    bool regularize;
};

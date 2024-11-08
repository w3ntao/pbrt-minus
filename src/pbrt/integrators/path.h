#pragma once

#include "pbrt/util/macro.h"

class BSDF;
class Sampler;
class ParameterDictionary;
struct IntegratorBase;

class PathIntegrator {
  public:
    static const PathIntegrator *create(const ParameterDictionary &parameters,
                                        const IntegratorBase *integrator_base,
                                        std::vector<void *> &gpu_dynamic_pointers);

    void init(const IntegratorBase *_base, uint _max_depth);

    PBRT_GPU
    static SampledSpectrum eval_li(const Ray &primary_ray, SampledWavelengths &lambda,
                                   const IntegratorBase *base, Sampler *sampler, uint max_depth);

    PBRT_GPU
    static inline SampledSpectrum sample_ld(const SurfaceInteraction &intr, const BSDF *bsdf,
                                            SampledWavelengths &lambda, const IntegratorBase *base,
                                            Sampler *sampler);

    PBRT_GPU
    SampledSpectrum li(const Ray &primary_ray, SampledWavelengths &lambda, Sampler *sampler) const;

  private:
    const IntegratorBase *base;
    uint max_depth;
};

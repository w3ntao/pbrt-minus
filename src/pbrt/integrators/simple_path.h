#pragma once

#include "pbrt/light_samplers/uniform_light_sampler.h"

class IntegratorBase;
class Sampler;

class SimplePathIntegrator {
  public:
    static const SimplePathIntegrator *create(const ParameterDictionary &parameters,
                                              const IntegratorBase *integrator_base,
                                              std::vector<void *> &gpu_dynamic_pointers);

    void init(const IntegratorBase *_base, uint _max_depth);

    PBRT_GPU
    SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler) const;

  private:
    const IntegratorBase *base;
    uint max_depth;
};

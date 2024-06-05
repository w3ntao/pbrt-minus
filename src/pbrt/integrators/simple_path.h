#pragma once

#include "pbrt/light_samplers/uniform_light_sampler.h"

class IntegratorBase;
class Sampler;

class SimplePathIntegrator {
  public:
    void init(const IntegratorBase *_base, uint _max_depth);

    PBRT_GPU
    SampledSpectrum li(const DifferentialRay &ray, SampledWavelengths &lambda, Sampler *sampler);

  private:
    const IntegratorBase *base;
    uint max_depth;
};

#pragma once

#include "pbrt/base/spectrum.h"
#include "pbrt/util/sampling.h"
#include "pbrt/euclidean_space/frame.h"

class DifferentialRay;
class IntegratorBase;
class Sampler;

class RandomWalkIntegrator {
  public:
    void init(const IntegratorBase *_base, uint _max_depth);

    PBRT_GPU SampledSpectrum li(const DifferentialRay &ray, SampledWavelengths &lambda,
                                Sampler *sampler) const {
        return li_random_walk(ray, lambda, sampler, 0);
    }

  private:
    PBRT_GPU
    SampledSpectrum li_random_walk(const DifferentialRay &ray, SampledWavelengths &lambda,
                                   Sampler *sampler, uint depth) const;

    const IntegratorBase *base;
    uint max_depth;
};

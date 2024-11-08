#pragma once

#include "pbrt/util/macro.h"
#include <vector>

class BSDF;
class ParameterDictionary;
class Ray;
class SampledSpectrum;
class SampledWavelengths;
class Sampler;
class SurfaceInteraction;

struct IntegratorBase;

class BDPTIntegrator {
  public:
    static const BDPTIntegrator *create(const ParameterDictionary &parameters,
                                        const IntegratorBase *integrator_base,
                                        std::vector<void *> &gpu_dynamic_pointers);

    void init(const IntegratorBase *_base, uint _max_depth);

    PBRT_GPU
    SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler) const;

  private:
    const IntegratorBase *base;
};

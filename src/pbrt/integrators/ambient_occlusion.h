#pragma once

#include <pbrt/spectrum_util/sampled_spectrum.h>

class ParameterDictionary;
class Sampler;

struct IntegratorBase;

class AmbientOcclusionIntegrator {
  public:
    AmbientOcclusionIntegrator(const ParameterDictionary &parameters,
                               const IntegratorBase *integrator_base);

    PBRT_CPU_GPU
    SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler) const;

  private:
    const IntegratorBase *base = nullptr;
    const Spectrum *illuminant_spectrum = nullptr;
    Real illuminant_scale = NAN;
};

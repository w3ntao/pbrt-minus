#pragma once

#include <pbrt/accelerator/hlbvh.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/ray.h>
#include <pbrt/base/sampler.h>

class GPUMemoryAllocator;
class ParameterDictionary;

struct IntegratorBase;

class AmbientOcclusionIntegrator {
  public:
    static const AmbientOcclusionIntegrator *create(const ParameterDictionary &parameters,
                                                    const IntegratorBase *integrator_base,
                                                    GPUMemoryAllocator &allocator);

    void init(const IntegratorBase *_base, const Spectrum *_illuminant_spectrum,
              const FloatType _illuminant_scale) {
        base = _base;
        illuminant_spectrum = _illuminant_spectrum;
        illuminant_scale = _illuminant_scale;
    }

    PBRT_CPU_GPU
    SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler) const;

  private:
    const IntegratorBase *base;
    const Spectrum *illuminant_spectrum;
    FloatType illuminant_scale;
};

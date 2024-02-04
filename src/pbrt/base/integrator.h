#pragma once

#include "pbrt/spectra/sampled_spectrum.h"
#include "pbrt/base/rgb.h"
#include "pbrt/base/ray.h"
#include "pbrt/base/aggregate.h"
#include "pbrt/base/sampler.h"

class Integrator {
  public:
    PBRT_GPU virtual ~Integrator() {}

    PBRT_GPU virtual SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda,
                                        const Aggregate *aggregate, Sampler &sampler) const = 0;
};

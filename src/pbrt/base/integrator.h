#pragma once

#include "pbrt/base/ray.h"
#include "pbrt/accelerator/hlbvh.h"
#include "pbrt/base/sampler.h"
#include "pbrt/spectra/sampled_spectrum.h"

class Integrator {
  public:
    PBRT_GPU virtual ~Integrator() {}

    PBRT_GPU virtual SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda,
                                        const HLBVH *bvh, Sampler &sampler) const = 0;
};

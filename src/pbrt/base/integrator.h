#pragma once

#include "pbrt/base/rgb.h"
#include "pbrt/base/ray.h"
#include "pbrt/base/aggregate.h"
#include "pbrt/base/sampler.h"

class Integrator {
  public:
    PBRT_GPU virtual ~Integrator() {}

    PBRT_GPU virtual RGB li(const Ray &ray, const Aggregate *aggregate, Sampler *sampler) const = 0;
};

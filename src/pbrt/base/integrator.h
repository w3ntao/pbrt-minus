#pragma once

#include <curand_kernel.h>

#include "pbrt/base/rgb.h"
#include "pbrt/base/ray.h"
#include "pbrt/base/aggregate.h"

class Integrator {
  public:
    PBRT_GPU virtual ~Integrator() {}

    PBRT_GPU virtual RGB li(const Ray &ray, const Aggregate *aggregate,
                            curandState *local_rand_state) const = 0;
};

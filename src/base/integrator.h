#pragma once

#include "base/color.h"
#include "base/ray.h"
#include "base/world.h"
#include <curand_kernel.h>

class Integrator {
  public:
    PBRT_GPU virtual ~Integrator() {}

    PBRT_GPU virtual Color get_radiance(const Ray &ray, const World *world,
                                        curandState *local_rand_state) const = 0;
};

#pragma once

#include "base/color.h"

class Integrator {
    public:
        __device__ virtual ~Integrator() {}

        __device__ virtual Color get_radiance(const Ray &ray, const World *const *world,
                                              curandState *local_rand_state) const = 0;
};

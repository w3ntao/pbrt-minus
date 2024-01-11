#pragma once

#include "base/color.h"

class Integrator {
    public:
        PBRT_GPU virtual ~Integrator() {}

        PBRT_GPU virtual Color get_radiance(const Ray &ray, const World *const *world,
                                            curandState *local_rand_state) const = 0;
};

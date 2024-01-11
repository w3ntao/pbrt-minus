//
// Created by wentao on 4/13/23.
//

#ifndef CUDA_RAY_TRACER_INTEGRATOR_H
#define CUDA_RAY_TRACER_INTEGRATOR_H

#include "base/color.h"

class Integrator {
    public:
        __device__ virtual ~Integrator() {}

        __device__ virtual Color get_radiance(const Ray &ray, const World *const *world,
                                              curandState *local_rand_state) const = 0;
};

#endif // CUDA_RAY_TRACER_INTEGRATOR_H

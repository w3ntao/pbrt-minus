#pragma once

#include "base/integrator.h"

class PathIntegrator : public Integrator {
  public:
    ~PathIntegrator() override = default;

    PBRT_GPU Color get_radiance(const Ray &ray, const World *world,
                                curandState *local_rand_state) const override {

        printf("PathIntegrator not implemented\n");
        asm("trap;");

        return Color(0.0, 0.0, 0.0); // exceeded recursion
    }
};

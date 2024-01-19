#pragma once

#include "base/integrator.h"

class SurfaceNormalIntegrator : public Integrator {
  public:
    ~SurfaceNormalIntegrator() override = default;

    PBRT_GPU Color get_radiance(const Ray &ray, const World *world,
                                curandState *local_rand_state) const override {

        // return Color(1.0, 1.0, 1.0);

        const auto shape_intersection = world->intersect(ray);

        if (!shape_intersection) {
            return Color(0.0, 0.0, 0.0);
        }

        const Normal3f normal = shape_intersection->interation.n;

        const Vector3f n = normal.to_vector3().softmax();

        return Color(n.x, n.y, n.z);
    }
};

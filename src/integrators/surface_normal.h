#pragma once

#include <limits>
#include "base/integrator.h"

class SurfaceNormalIntegrator : public Integrator {
    public:
        ~SurfaceNormalIntegrator() override = default;

        PBRT_GPU Color get_radiance(const Ray &ray, const World *world,
                                    curandState *local_rand_state) const override {
            Intersection intersection;
            if (!world->intersect(intersection, ray, 0.001f, std::numeric_limits<double>::max())) {
                return Color(0.0, 0.0, 0.0);
            }

            const Vector3f n = Vector3f(intersection.n).softmax();
            return Color(n.x, n.y, n.z);
        }
};

#pragma once

#include <limits>
#include "base/integrator.h"

class SurfaceNormalIntegrator : public Integrator {
    public:
        PBRT_GPU SurfaceNormalIntegrator() = default;

        ~SurfaceNormalIntegrator() override = default;

        PBRT_GPU Color get_radiance(const Ray &ray, const World *const *world,
                                    curandState *local_rand_state) const override {
            Intersection intersection;
            if (!(*world)->intersect(intersection, ray, 0.001f, std::numeric_limits<double>::max())) {
                return Color(0.0, 0.0, 0.0);
            }

            const Vector3 n = Vector3(intersection.n).softmax();
            return Color(n.x, n.y, n.z);
        }
};

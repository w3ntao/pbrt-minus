#pragma once

#include <limits>
#include "base/integrator.h"

class PathIntegrator : public Integrator {
  public:
    ~PathIntegrator() override = default;

    PBRT_GPU Color get_radiance(const Ray &ray, const World *world,
                                curandState *local_rand_state) const override {
        Ray current_ray = ray;
        Color current_attenuation = Color(1.0, 1.0, 1.0);
        for (int i = 0; i < 50; i++) {

            //TODO: progrss 2024/01/17: rewrite PathIntegrator

            /*
            Intersection intersection;
            if (world->intersect(intersection, current_ray, 0.001f,
                                 std::numeric_limits<double>::max())) {
                Ray scattered_ray;
                Color attenuation;
                if (!intersection.material_ptr->scatter(current_ray, intersection, attenuation,
                                                        scattered_ray, local_rand_state)) {
                    return Color(0.0, 0.0, 0.0);
                }

                current_attenuation *= attenuation;
                current_ray = scattered_ray;
                continue;
            }
            */

            double t = 0.5f * (current_ray.d.normalize().y + 1.0f);
            Vector3f c = (1.0f - t) * Vector3f(1.0, 1.0, 1.0) + t * Vector3f(0.5, 0.7, 1.0);
            return current_attenuation * Color(c.x, c.y, c.z);
        }

        return Color(0.0, 0.0, 0.0); // exceeded recursion
    }
};

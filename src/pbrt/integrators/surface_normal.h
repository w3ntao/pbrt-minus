#pragma once

#include "pbrt/base/integrator.h"
#include "pbrt/util/sampling.h"

class SurfaceNormalIntegrator : public Integrator {
  public:
    ~SurfaceNormalIntegrator() override = default;

    PBRT_GPU RGB li(const Ray &ray, const Aggregate *aggregate, Sampler *sampler) const override {
        const auto shape_intersection = aggregate->intersect(ray);

        if (!shape_intersection) {
            return RGB(0.0, 0.0, 0.0);
        }

        Vector3f normal = shape_intersection->interation.n.to_vector3();
        normal = normal.face_forward(-ray.d);

        const Vector3f n = normal.softmax();
        return RGB(n.x, n.y, n.z);
    }
};

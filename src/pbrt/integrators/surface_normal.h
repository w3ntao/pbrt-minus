#pragma once

#include "pbrt/base/integrator.h"
#include "pbrt/util/sampling.h"

class SurfaceNormalIntegrator : public Integrator {
  public:
    PBRT_GPU SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda,
                                const Aggregate *aggregate, Sampler &sampler) const override {
        // TODO: li() for SurfaceNormalIntegrator is not implemented

        return SampledSpectrum(0);

        /*
        if (!shape_intersection) {
            return RGB(0.0, 0.0, 0.0);
        }

        Vector3f normal = shape_intersection->interation.n.to_vector3();
        normal = normal.face_forward(-ray.d);

        const Vector3f n = normal.softmax();
        return RGB(n.x, n.y, n.z);
        */
    }
};

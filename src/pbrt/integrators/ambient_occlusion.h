#pragma once

#include "pbrt/base/differential_ray.h"
#include "pbrt/base/integrator.h"
#include "pbrt/util/sampling.h"
#include "pbrt/euclidean_space/frame.h"

class AmbientOcclusionIntegrator : public Integrator {

    ~AmbientOcclusionIntegrator() override = default;

    PBRT_GPU RGB li(const Ray &ray, const Aggregate *aggregate, Sampler *sampler) const override {
        const auto shape_intersection = aggregate->intersect(ray);
        if (!shape_intersection) {
            return RGB(0.0, 0.0, 0.0);
        }

        const SurfaceInteraction &isect = shape_intersection->interation;

        auto normal = isect.n.to_vector3().face_forward(-ray.d);

        // return RGB(normal.dot(-ray.d));

        /*
        auto softmax_n = normal.softmax();
        return RGB(softmax_n.x, softmax_n.y, softmax_n.z);
        */

        auto u = sampler->get_2d();
        auto local_wi = sample_cosine_hemisphere(u);
        auto pdf = cosine_hemisphere_pdf(std::abs(local_wi.z));

        if (pdf == 0.0) {
            return RGB(0.0, 0.0, 0.0);
        }

        // return RGB(normal.dot(-ray.d));

        auto frame = Frame::from_z(normal);
        auto wi = frame.from_local(local_wi);

        /*
        auto softmax = wi.softmax();
        return RGB(softmax.x, softmax.y, softmax.z);
        */

        // this part is fine
        // TODO: progress 2024/01/30 wentao debugging AmbientOcclusionIntegrator

        // Divide by PI so that fully visible is one.
        auto spawned_ray = isect.spawn_ray(wi);
        if (!aggregate->fast_intersect(spawned_ray, Infinity)) {
            const auto grey = normal.dot(wi) / (compute_pi() * pdf);
            return RGB(grey);
        }

        return RGB(0.0, 0.0, 0.0);
    }
};

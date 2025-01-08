#pragma once

#include "pbrt/euclidean_space/transform.h"
#include "pbrt/util/macro.h"
#include <vector>

class ShapeSample;
class ShapeSampleContext;
class ParameterDictionary;

class Disk {
  public:
    static const Disk *create(const Transform &render_from_object,
                              const Transform &object_from_render, bool reverse_orientation,
                              const ParameterDictionary &parameters,
                              std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU
    FloatType area() const {
        return 0.5 * phi_max * (sqr(radius) - sqr(inner_radius));
    }

    PBRT_CPU_GPU
    Bounds3f bounds() const {
        return render_from_object(
            Bounds3f(Point3f(-radius, -radius, height), Point3f(radius, radius, height)));
    }

    PBRT_GPU
    bool fast_intersect(const Ray &ray, FloatType t_max = Infinity) const {
        return basic_intersect(ray, t_max).has_value();
    }

    PBRT_GPU
    pbrt::optional<ShapeIntersection> intersect(const Ray &ray,
                                                     FloatType t_max = Infinity) const {
        auto isect = basic_intersect(ray, t_max);
        if (!isect) {
            return {};
        }

        SurfaceInteraction intr = interaction_from_intersection(*isect, -ray.d);
        return ShapeIntersection{intr, isect->t_hit};
    }

    PBRT_GPU
    FloatType pdf(const ShapeSampleContext &ctx, const Vector3f &wi) const;

    PBRT_GPU
    pbrt::optional<ShapeSample> sample(const Point2f &u) const;

    PBRT_GPU
    pbrt::optional<ShapeSample> sample(const ShapeSampleContext &ctx, const Point2f &u) const;

  private:
    // Disk Private Members
    Transform render_from_object;
    Transform object_from_render;
    bool reverse_orientation;
    bool transform_wwapsHandedness;

    FloatType height;
    FloatType radius;
    FloatType inner_radius;
    FloatType phi_max;

    PBRT_CPU_GPU
    pbrt::optional<QuadricIntersection> basic_intersect(const Ray &r, FloatType tMax) const;

    PBRT_CPU_GPU
    SurfaceInteraction interaction_from_intersection(const QuadricIntersection &isect,
                                                     const Vector3f &wo) const;
};

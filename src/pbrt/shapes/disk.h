#pragma once

#include <pbrt/euclidean_space/transform.h>
#include <pbrt/gpu/macro.h>

class GPUMemoryAllocator;
class ParameterDictionary;

struct ShapeSample;
struct ShapeSampleContext;

class Disk {
  public:
    Disk(const Transform &_render_from_object, const Transform &_object_from_render,
         bool _reverse_orientation, const ParameterDictionary &parameters);

    PBRT_CPU_GPU
    Real area() const {
        return 0.5 * phi_max * (sqr(radius) - sqr(inner_radius));
    }

    PBRT_CPU_GPU
    Bounds3f bounds() const {
        return render_from_object(
            Bounds3f(Point3f(-radius, -radius, height), Point3f(radius, radius, height)));
    }

    PBRT_CPU_GPU
    bool fast_intersect(const Ray &ray, Real t_max = Infinity) const {
        return basic_intersect(ray, t_max).has_value();
    }

    PBRT_CPU_GPU
    pbrt::optional<ShapeIntersection> intersect(const Ray &ray, Real t_max = Infinity) const {
        auto isect = basic_intersect(ray, t_max);
        if (!isect) {
            return {};
        }

        SurfaceInteraction intr = interaction_from_intersection(*isect, -ray.d);
        return ShapeIntersection{intr, isect->t_hit};
    }

    PBRT_CPU_GPU
    Real pdf(const ShapeSampleContext &ctx, const Vector3f &wi) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeSample> sample(const Point2f &u) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeSample> sample(const ShapeSampleContext &ctx, const Point2f &u) const;

  private:
    // Disk Private Members
    Transform render_from_object;
    Transform object_from_render;
    bool reverse_orientation = false;
    bool transform_swaps_handedness = false;

    Real height = NAN;
    Real radius = NAN;
    Real inner_radius = NAN;
    Real phi_max = NAN;

    PBRT_CPU_GPU
    pbrt::optional<QuadricIntersection> basic_intersect(const Ray &r, Real tMax) const;

    PBRT_CPU_GPU
    SurfaceInteraction interaction_from_intersection(const QuadricIntersection &isect,
                                                     const Vector3f &wo) const;
};

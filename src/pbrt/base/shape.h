#pragma once

#include <cuda/std/optional>
#include <vector>

#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/base/interaction.h"
#include "pbrt/base/ray.h"

class Sphere;
class Triangle;

class Transform;
class ParameterDict;

struct ShapeSampleContext {
    Point3fi pi;
    Normal3f n;
    Normal3f ns;

    PBRT_CPU_GPU
    Point3f p() const {
        return pi.to_point3f();
    }

    PBRT_CPU_GPU
    inline Point3f offset_ray_origin(const Vector3f &w) const {
        // Find vector _offset_ to corner of error bounds and compute initial _po_
        FloatType d = n.abs().dot(pi.error());

        Vector3f offset = d * n.to_vector3();
        if (n.dot(w) < 0) {
            offset = -offset;
        }

        Point3f po = pi.to_point3f() + offset;

        // Round offset point _po_ away from _p_
        for (int i = 0; i < 3; ++i) {
            if (offset[i] > 0) {
                po[i] = next_float_up(po[i]);

            } else if (offset[i] < 0) {
                po[i] = next_float_down(po[i]);
            }
        }

        return po;
    }

    PBRT_CPU_GPU
    inline Point3f offset_ray_origin(const Point3f &pt) const {
        return this->offset_ray_origin(pt - p());
    }
};

struct ShapeSample {
    Interaction interaction;
    FloatType pdf;
};

class Shape {
  public:
    enum class Type {
        triangle,
        sphere,
    };

    static const Shape *create_sphere(const Transform &render_from_object,
                                      const Transform &object_from_render, bool reverse_orientation,
                                      const ParameterDict &parameters,
                                      std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU
    void init(const Triangle *triangle);

    PBRT_CPU_GPU
    void init(const Sphere *sphere);

    PBRT_CPU_GPU
    Bounds3f bounds() const;

    PBRT_CPU_GPU
    FloatType area() const;

    PBRT_GPU
    bool fast_intersect(const Ray &ray, FloatType t_max) const;

    PBRT_GPU
    cuda::std::optional<ShapeIntersection> intersect(const Ray &ray, FloatType t_max) const;

    PBRT_GPU
    cuda::std::optional<ShapeSample> sample(const ShapeSampleContext &ctx, const Point2f u) const;

  private:
    Type type;
    const void *shape;
};

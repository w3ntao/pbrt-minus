#pragma once

#include <pbrt/base/interaction.h>
#include <pbrt/base/ray.h>
#include <pbrt/euclidean_space/bounds3.h>
#include <pbrt/gpu/macro.h>
#include <vector>

class Disk;
class GPUMemoryAllocator;
class Sphere;
class Triangle;
class Transform;
class ParameterDictionary;

struct ShapeSampleContext {
    Point3fi pi;
    Normal3f n;
    Normal3f ns;

    PBRT_CPU_GPU
    ShapeSampleContext(const Point3fi &_pi, const Normal3f &_n, const Normal3f &_ns)
        : pi(_pi), n(_n), ns(_ns) {}

    PBRT_CPU_GPU
    Point3f p() const {
        return pi.to_point3f();
    }

    PBRT_CPU_GPU
    Point3f offset_ray_origin(const Vector3f &w) const {
        // Find vector _offset_ to corner of error bounds and compute initial _po_
        Real d = n.abs().dot(pi.error());

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
    Point3f offset_ray_origin(const Point3f &pt) const {
        return this->offset_ray_origin(pt - p());
    }

    PBRT_CPU_GPU
    Ray spawn_ray(const Vector3f &w) const {
        // Note: doesn't set medium, but that's fine, since this is only
        // used by shapes to see if ray would have intersected them
        return Ray(this->offset_ray_origin(w), w);
    }
};

struct ShapeSample {
    Interaction interaction;
    Real pdf;
};

class Shape {
  public:
    enum class Type {
        disk,
        sphere,
        triangle,
    };

    PBRT_CPU_GPU
    explicit Shape(const Disk *disk) : type(Type::disk), ptr(disk) {}

    PBRT_CPU_GPU
    explicit Shape(const Triangle *triangle) : type(Type::triangle), ptr(triangle) {}

    PBRT_CPU_GPU
    explicit Shape(const Sphere *sphere) : type(Type::sphere), ptr(sphere) {}

    static std::pair<const Shape *, int>
    create(const std::string &type_of_shape, const Transform &render_from_object,
           const Transform &object_from_render, bool reverse_orientation,
           const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Bounds3f bounds() const;

    PBRT_CPU_GPU
    Real area() const;

    PBRT_CPU_GPU
    bool fast_intersect(const Ray &ray, Real t_max) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeIntersection> intersect(const Ray &ray, Real t_max = Infinity) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeSample> sample(Point2f u) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeSample> sample(const ShapeSampleContext &ctx, const Point2f &u) const;

    PBRT_CPU_GPU
    Real pdf(const Interaction &in) const;

    PBRT_CPU_GPU
    Real pdf(const ShapeSampleContext &ctx, const Vector3f &wi) const;

  private:
    Type type;
    const void *ptr = nullptr;
};

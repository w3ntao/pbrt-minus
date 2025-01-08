#pragma once

#include "pbrt/base/interaction.h"
#include "pbrt/base/ray.h"
#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/shapes/triangle_mesh.h"
#include "pbrt/util/sampling.h"

class ShapeSampleContext;
class ShapeSample;
struct ShapeIntersection;

class Triangle {
  public:
    struct TriangleIntersection {
        FloatType b0, b1, b2;
        FloatType t;

        PBRT_CPU_GPU
        TriangleIntersection(FloatType _b0, FloatType _b1, FloatType _b2, FloatType _t)
            : b0(_b0), b1(_b1), b2(_b2), t(_t) {}
    };

    PBRT_CPU_GPU
    static FloatType spherical_triangle_area(const Vector3f a, const Vector3f b, const Vector3f c) {
        return std::abs(2.0 * std::atan2(a.dot(b.cross(c)), 1 + a.dot(b) + a.dot(c) + b.dot(c)));
    }

    PBRT_CPU_GPU
    void init(int idx, const TriangleMesh *_mesh) {
        triangle_idx = idx;
        mesh = _mesh;
    }

    PBRT_CPU_GPU
    Bounds3f bounds() const {
        Point3f points[3];
        get_points(points);

        return Bounds3f(points, 3);
    }

    PBRT_CPU_GPU
    FloatType area() const {
        Point3f p[3];
        get_points(p);

        return (p[1] - p[0]).cross((p[2] - p[0])).length() * 0.5;
    }

    PBRT_CPU_GPU
    bool fast_intersect(const Ray &ray, FloatType t_max) const {
        Point3f points[3];
        get_points(points);

        return intersect_triangle(ray, t_max, points[0], points[1], points[2]).has_value();
    }

    PBRT_CPU_GPU
    pbrt::optional<ShapeIntersection> intersect(const Ray &ray, FloatType t_max) const;

    PBRT_CPU_GPU
    FloatType pdf(const ShapeSampleContext &ctx, const Vector3f &wi) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeSample> sample(Point2f u) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeSample> sample(const ShapeSampleContext &ctx, Point2f u) const;

  private:
    int triangle_idx;
    const TriangleMesh *mesh;
    static constexpr FloatType MinSphericalSampleArea = 3e-4;
    static constexpr FloatType MaxSphericalSampleArea = 6.22;

    PBRT_CPU_GPU
    void get_points(Point3f p[3]) const {
        const int *v = &(mesh->vertex_indices[3 * triangle_idx]);
        for (uint idx = 0; idx < 3; ++idx) {
            p[idx] = mesh->p[v[idx]];
        }
    }

    PBRT_CPU_GPU
    FloatType solid_angle(const Point3f p) const {
        // Get triangle vertices in _p0_, _p1_, and _p2_
        Point3f points[3];
        get_points(points);

        return spherical_triangle_area((points[0] - p).normalize(), (points[1] - p).normalize(),
                                       (points[2] - p).normalize());
    }

    PBRT_CPU_GPU
    pbrt::optional<Triangle::TriangleIntersection>
    intersect_triangle(const Ray &ray, FloatType t_max, const Point3f &p0, const Point3f &p1,
                       const Point3f &p2) const;

    PBRT_CPU_GPU
    SurfaceInteraction interaction_from_intersection(const TriangleIntersection &ti,
                                                     const Vector3f &wo) const;
};

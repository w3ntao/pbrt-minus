#pragma once

#include <pbrt/base/interaction.h>
#include <pbrt/base/ray.h>
#include <pbrt/euclidean_space/bounds3.h>
#include <pbrt/shapes/triangle_mesh.h>
#include <pbrt/util/sampling.h>

struct ShapeIntersection;
struct ShapeSample;
struct ShapeSampleContext;

class Triangle {
  public:
    struct TriangleIntersection {
        Real b0, b1, b2;
        Real t;

        PBRT_CPU_GPU
        TriangleIntersection(Real _b0, Real _b1, Real _b2, Real _t)
            : b0(_b0), b1(_b1), b2(_b2), t(_t) {}
    };

    PBRT_CPU_GPU
    static Real spherical_triangle_area(const Vector3f a, const Vector3f b, const Vector3f c) {
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
    Real area() const {
        const auto [p0, p1, p2] = get_points();
        return (p1 - p0).cross((p2 - p0)).length() * 0.5;
    }

    PBRT_CPU_GPU
    bool fast_intersect(const Ray &ray, Real t_max) const {
        const auto [p0, p1, p2] = get_points();

        return intersect_triangle(ray, t_max, p0, p1, p2).has_value();
    }

    PBRT_CPU_GPU
    pbrt::optional<ShapeIntersection> intersect(const Ray &ray, Real t_max) const;

    PBRT_CPU_GPU
    Real pdf(const Interaction &in) const;

    PBRT_CPU_GPU
    Real pdf(const ShapeSampleContext &ctx, const Vector3f &wi) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeSample> sample(Point2f u) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeSample> sample(const ShapeSampleContext &ctx, Point2f u) const;

  private:
    int triangle_idx;
    const TriangleMesh *mesh;
    static constexpr Real MinSphericalSampleArea = 3e-4;
    static constexpr Real MaxSphericalSampleArea = 6.22;

    PBRT_CPU_GPU
    void get_points(Point3f p[3]) const {
        const int *v = &(mesh->vertex_indices[3 * triangle_idx]);
        for (int idx = 0; idx < 3; ++idx) {
            p[idx] = mesh->p[v[idx]];
        }
    }

    PBRT_CPU_GPU
    cuda::std::tuple<Point3f, Point3f, Point3f> get_points() const {
        const int *v = &(mesh->vertex_indices[3 * triangle_idx]);

        return {mesh->p[v[0]], mesh->p[v[1]], mesh->p[v[2]]};
    }

    PBRT_CPU_GPU
    Real solid_angle(const Point3f p) const {
        const auto [p0, p1, p2] = get_points();

        return spherical_triangle_area((p0 - p).normalize(), (p1 - p).normalize(),
                                       (p2 - p).normalize());
    }

    PBRT_CPU_GPU
    pbrt::optional<Triangle::TriangleIntersection> intersect_triangle(const Ray &ray, Real t_max,
                                                                      const Point3f &p0,
                                                                      const Point3f &p1,
                                                                      const Point3f &p2) const;

    PBRT_CPU_GPU
    SurfaceInteraction interaction_from_intersection(const TriangleIntersection &ti,
                                                     const Vector3f &wo) const;
};

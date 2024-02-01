#pragma once

#include "pbrt/base/shape.h"
#include "pbrt/euclidean_space/vector2.h"
#include "pbrt/util/accurate_arithmetic.h"
#include "pbrt/shapes/triangle_mesh.h"

struct TriangleIntersection {
    double b0, b1, b2;
    double t;

    PBRT_CPU_GPU TriangleIntersection(double _b0, double _b1, double _b2, double _t)
        : b0(_b0), b1(_b1), b2(_b2), t(_t) {}
};

class Triangle final : public Shape {
  public:
    const int triangle_idx;
    const TriangleMesh *mesh;

    PBRT_GPU Triangle(int _idx, const TriangleMesh *_mesh) : triangle_idx(_idx), mesh(_mesh) {}

    PBRT_GPU ~Triangle() override {
        if (triangle_idx == 0) {
            delete mesh;
        }
    }

    PBRT_GPU bool fast_intersect(const Ray &ray, double t_max) const override {
        auto points = get_points();
        auto p0 = points[0];
        auto p1 = points[1];
        auto p2 = points[2];

        return intersect_triangle(ray, t_max, p0, p1, p2).has_value();
    }

    PBRT_GPU std::optional<ShapeIntersection> intersect(const Ray &ray,
                                                        double t_max) const override {
        auto points = get_points();
        auto p0 = points[0];
        auto p1 = points[1];
        auto p2 = points[2];

        std::optional<TriangleIntersection> tri_intersection =
            intersect_triangle(ray, t_max, p0, p1, p2);
        if (!tri_intersection) {
            return {};
        }

        SurfaceInteraction si = interaction_from_intersection(tri_intersection.value(), -ray.d);

        return ShapeIntersection(si, tri_intersection->t);
    }

  private:
    PBRT_GPU
    std::array<Point3f, 3> get_points() const {
        const int *v = &(mesh->vertex_indices[3 * triangle_idx]);
        const Point3f p0 = mesh->p[v[0]];
        const Point3f p1 = mesh->p[v[1]];
        const Point3f p2 = mesh->p[v[2]];

        return {p0, p1, p2};
    }

    PBRT_GPU std::optional<TriangleIntersection> intersect_triangle(const Ray &ray, double t_max,
                                                                    const Point3f &p0,
                                                                    const Point3f &p1,
                                                                    const Point3f &p2) const {
        // Return no intersection if triangle is degenerate
        if ((p2 - p0).cross(p1 - p0).squared_length() == 0.0) {
            return {};
        }

        // Transform triangle vertices to ray coordinate space
        // Translate vertices based on ray origin
        Point3f p0t = p0 - ray.o.to_vector3();
        Point3f p1t = p1 - ray.o.to_vector3();
        Point3f p2t = p2 - ray.o.to_vector3();

        // Permute components of triangle vertices and ray direction

        int kz = ray.d.abs().max_component_index();
        int kx = (kz + 1) % 3;
        int ky = (kz + 2) % 3;

        Vector3f d = ray.d.permute({kx, ky, kz});
        p0t = p0t.permute({kx, ky, kz});
        p1t = p1t.permute({kx, ky, kz});
        p2t = p2t.permute({kx, ky, kz});

        // Apply shear transformation to translated vertex positions
        double Sx = -d.x / d.z;
        double Sy = -d.y / d.z;
        double Sz = 1 / d.z;
        p0t.x += Sx * p0t.z;
        p0t.y += Sy * p0t.z;
        p1t.x += Sx * p1t.z;
        p1t.y += Sy * p1t.z;
        p2t.x += Sx * p2t.z;
        p2t.y += Sy * p2t.z;

        // Compute edge function coefficients _e0_, _e1_, and _e2_
        double e0 = difference_of_products(p1t.x, p2t.y, p1t.y, p2t.x);
        double e1 = difference_of_products(p2t.x, p0t.y, p2t.y, p0t.x);
        double e2 = difference_of_products(p0t.x, p1t.y, p0t.y, p1t.x);

        // Perform triangle edge and determinant tests
        if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0)) {
            return {};
        }

        double det = e0 + e1 + e2;
        if (det == 0) {
            return {};
        }

        // Compute scaled hit distance to triangle and test against ray $t$ range
        p0t.z *= Sz;
        p1t.z *= Sz;
        p2t.z *= Sz;
        double tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
        if (det < 0 && (tScaled >= 0 || tScaled < t_max * det)) {
            return {};
        }

        if (det > 0 && (tScaled <= 0 || tScaled > t_max * det)) {
            return {};
        }

        // Compute barycentric coordinates and $t$ value for triangle intersection
        double invDet = 1 / det;
        double b0 = e0 * invDet, b1 = e1 * invDet, b2 = e2 * invDet;
        double t = tScaled * invDet;

        // Ensure that computed triangle $t$ is conservatively greater than zero
        // Compute $\delta_z$ term for triangle $t$ error bounds
        // double maxZt = MaxComponentValue(Abs(Vector3f(p0t.z, p1t.z, p2t.z)));

        double maxZt = Vector3f(p0t.z, p1t.z, p2t.z).abs().max_component_value();
        double deltaZ = gamma(3) * maxZt;

        // Compute $\delta_x$ and $\delta_y$ terms for triangle $t$ error bounds
        double maxXt = Vector3f(p0t.x, p1t.x, p2t.x).abs().max_component_value();
        double maxYt = Vector3f(p0t.y, p1t.y, p2t.y).abs().max_component_value();

        double deltaX = gamma(5) * (maxXt + maxZt);
        double deltaY = gamma(5) * (maxYt + maxZt);

        // Compute $\delta_e$ term for triangle $t$ error bounds
        double deltaE = 2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

        // Compute $\delta_t$ term for triangle $t$ error bounds and check _t_
        double maxE = Vector3f(e0, e1, e2).abs().max_component_value();
        double deltaT =
            3 * (gamma(3) * maxE * maxZt + deltaE * maxZt + deltaZ * maxE) * std::abs(invDet);
        if (t <= deltaT) {
            return {};
        }

        // Return _TriangleIntersection_ for intersection
        return TriangleIntersection(b0, b1, b2, t);
    }

    PBRT_GPU SurfaceInteraction interaction_from_intersection(const TriangleIntersection &ti,
                                                              const Vector3f &wo) const {
        const int *v = &(mesh->vertex_indices[3 * triangle_idx]);
        const Point3f p0 = mesh->p[v[0]];
        const Point3f p1 = mesh->p[v[1]];
        const Point3f p2 = mesh->p[v[2]];

        // Compute triangle partial derivatives
        // Compute deltas and matrix determinant for triangle partial derivatives
        // Get triangle texture coordinates in _uv_ array
        std::array<Point2f, 3> uv =
            mesh->uv ? std::array<Point2f, 3>({mesh->uv[v[0]], mesh->uv[v[1]], mesh->uv[v[2]]})
                     : std::array<Point2f, 3>({Point2f(0, 0), Point2f(1, 0), Point2f(1, 1)});

        Vector2f duv02 = uv[0] - uv[2], duv12 = uv[1] - uv[2];
        Vector3f dp02 = p0 - p2;
        Vector3f dp12 = p1 - p2;
        double determinant = difference_of_products(duv02[0], duv12[1], duv02[1], duv12[0]);

        Vector3f dpdu, dpdv;
        bool degenerateUV = std::abs(determinant) < 1e-9f;
        if (!degenerateUV) {
            // Compute triangle $\dpdu$ and $\dpdv$ via matrix inversion
            double invdet = 1 / determinant;
            dpdu = difference_of_products(duv12[1], dp02, duv02[1], dp12) * invdet;
            dpdv = difference_of_products(duv02[0], dp12, duv12[0], dp02) * invdet;
        }
        // Handle degenerate triangle $(u,v)$ parameterization or partial derivatives
        if (degenerateUV || dpdu.cross(dpdv).squared_length() == 0) {
            Vector3f ng = (p2 - p0).cross(p1 - p0);
            if (ng.squared_length() == 0) {
                ng = (p2 - p0).cross(p1 - p0);
            }
            ng.normalize().coordinate_system(&dpdu, &dpdv);
        }

        // Interpolate $(u,v)$ parametric coordinates and hit point
        Point3f pHit = ti.b0 * p0 + ti.b1 * p1 + ti.b2 * p2;
        Point2f uvHit = ti.b0 * uv[0] + ti.b1 * uv[1] + ti.b2 * uv[2];

        bool flipNormal = mesh->reverse_orientation ^ mesh->transformSwapsHandedness;
        // Compute error bounds _pError_ for triangle intersection
        Point3f pAbsSum = (ti.b0 * p0).abs() + (ti.b1 * p1).abs() + (ti.b2 * p2).abs();
        Vector3f pError = gamma(7) * pAbsSum.to_vector3();

        SurfaceInteraction isect(Point3fi(pHit, pError), uvHit, wo, dpdu, dpdv, Normal3f(),
                                 Normal3f(), flipNormal);

        isect.faceIndex = mesh->faceIndices ? mesh->faceIndices[triangle_idx] : 0;

        // Set final surface normal and shading geometry for triangle
        // Override surface normal in _isect_ for triangle
        isect.n = Normal3f(dp02.cross(dp12).normalize());
        isect.shading.n = isect.n;

        if (mesh->reverse_orientation ^ mesh->transformSwapsHandedness) {
            isect.n = isect.shading.n = -isect.n;
        }

        if (mesh->n || mesh->s) {
            asm("trap;");
            // TODO: mesh->n and mesh->s are not implemented
        }

        return isect;
    }
};

#pragma once

#include <optional>

#include "pbrt/util/macro.h"
#include "pbrt/base/ray.h"
#include "pbrt/base/interaction.h"
#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/shapes/triangle_mesh.h"

class Triangle {
  public:
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

    PBRT_GPU bool fast_intersect(const Ray &ray, FloatType t_max) const {
        Point3f points[3];
        get_points(points);

        return intersect_triangle(ray, t_max, points[0], points[1], points[2]).has_value();
    }

    PBRT_GPU std::optional<ShapeIntersection> intersect(const Ray &ray, FloatType t_max) const {
        Point3f points[3];
        get_points(points);

        std::optional<TriangleIntersection> tri_intersection =
            intersect_triangle(ray, t_max, points[0], points[1], points[2]);
        if (!tri_intersection) {
            return {};
        }

        SurfaceInteraction si = interaction_from_intersection(tri_intersection.value(), -ray.d);

        return ShapeIntersection(si, tri_intersection->t);
    }

  private:
    struct TriangleIntersection {
        FloatType b0, b1, b2;
        FloatType t;

        PBRT_CPU_GPU TriangleIntersection(FloatType _b0, FloatType _b1, FloatType _b2, FloatType _t)
            : b0(_b0), b1(_b1), b2(_b2), t(_t) {}
    };

    int triangle_idx;
    const TriangleMesh *mesh;

    PBRT_CPU_GPU

    void get_points(Point3f p[3]) const {
        const int *v = &(mesh->vertex_indices[3 * triangle_idx]);
        for (uint idx = 0; idx < 3; ++idx) {
            p[idx] = mesh->p[v[idx]];
        }
    }

    PBRT_GPU std::optional<TriangleIntersection> intersect_triangle(const Ray &ray, FloatType t_max,
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
        uint8_t kz = ray.d.abs().max_component_index();
        uint8_t kx = (kz + 1) % 3;
        uint8_t ky = (kz + 2) % 3;

        uint8_t permuted_idx[3] = {kx, ky, kz};
        Vector3f d = ray.d.permute(permuted_idx);

        p0t = p0t.permute(permuted_idx);
        p1t = p1t.permute(permuted_idx);
        p2t = p2t.permute(permuted_idx);

        // Apply shear transformation to translated vertex positions
        FloatType Sx = -d.x / d.z;
        FloatType Sy = -d.y / d.z;
        FloatType Sz = 1 / d.z;
        p0t.x += Sx * p0t.z;
        p0t.y += Sy * p0t.z;
        p1t.x += Sx * p1t.z;
        p1t.y += Sy * p1t.z;
        p2t.x += Sx * p2t.z;
        p2t.y += Sy * p2t.z;

        // Compute edge function coefficients _e0_, _e1_, and _e2_
        FloatType e0 = difference_of_products(p1t.x, p2t.y, p1t.y, p2t.x);
        FloatType e1 = difference_of_products(p2t.x, p0t.y, p2t.y, p0t.x);
        FloatType e2 = difference_of_products(p0t.x, p1t.y, p0t.y, p1t.x);

        // Fall back to FloatType-precision test at triangle edges
        if (sizeof(FloatType) == sizeof(float) && (e0 == 0.0f || e1 == 0.0f || e2 == 0.0f)) {
            FloatType p2txp1ty = (FloatType)p2t.x * (FloatType)p1t.y;
            FloatType p2typ1tx = (FloatType)p2t.y * (FloatType)p1t.x;
            e0 = (float)(p2typ1tx - p2txp1ty);
            FloatType p0txp2ty = (FloatType)p0t.x * (FloatType)p2t.y;
            FloatType p0typ2tx = (FloatType)p0t.y * (FloatType)p2t.x;
            e1 = (float)(p0typ2tx - p0txp2ty);
            FloatType p1txp0ty = (FloatType)p1t.x * (FloatType)p0t.y;
            FloatType p1typ0tx = (FloatType)p1t.y * (FloatType)p0t.x;
            e2 = (float)(p1typ0tx - p1txp0ty);
        }

        // Perform triangle edge and determinant tests
        if ((e0 < 0 || e1 < 0 || e2 < 0) && (e0 > 0 || e1 > 0 || e2 > 0)) {
            return {};
        }

        FloatType det = e0 + e1 + e2;
        if (det == 0) {
            return {};
        }

        // Compute scaled hit distance to triangle and test against ray $t$ range
        p0t.z *= Sz;
        p1t.z *= Sz;
        p2t.z *= Sz;
        FloatType tScaled = e0 * p0t.z + e1 * p1t.z + e2 * p2t.z;
        if (det < 0 && (tScaled >= 0 || tScaled < t_max * det)) {
            return {};
        }

        if (det > 0 && (tScaled <= 0 || tScaled > t_max * det)) {
            return {};
        }

        // Compute barycentric coordinates and $t$ value for triangle intersection
        FloatType invDet = 1 / det;
        FloatType b0 = e0 * invDet, b1 = e1 * invDet, b2 = e2 * invDet;
        FloatType t = tScaled * invDet;

        // Ensure that computed triangle $t$ is conservatively greater than zero
        // Compute $\delta_z$ term for triangle $t$ error bounds
        // FloatType maxZt = MaxComponentValue(Abs(Vector3f(p0t.z, p1t.z, p2t.z)));

        FloatType maxZt = Vector3f(p0t.z, p1t.z, p2t.z).abs().max_component_value();
        FloatType deltaZ = gamma(3) * maxZt;

        // Compute $\delta_x$ and $\delta_y$ terms for triangle $t$ error bounds
        FloatType maxXt = Vector3f(p0t.x, p1t.x, p2t.x).abs().max_component_value();
        FloatType maxYt = Vector3f(p0t.y, p1t.y, p2t.y).abs().max_component_value();

        FloatType deltaX = gamma(5) * (maxXt + maxZt);
        FloatType deltaY = gamma(5) * (maxYt + maxZt);

        // Compute $\delta_e$ term for triangle $t$ error bounds
        FloatType deltaE = 2 * (gamma(2) * maxXt * maxYt + deltaY * maxXt + deltaX * maxYt);

        // Compute $\delta_t$ term for triangle $t$ error bounds and check _t_
        FloatType maxE = Vector3f(e0, e1, e2).abs().max_component_value();
        FloatType deltaT =
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

        Point2f uv[3];
        if (mesh->uv) {
            for (uint idx = 0; idx < 3; ++idx) {
                uv[idx] = mesh->uv[v[idx]];
            }
        } else {
            uv[0] = Point2f(0, 0);
            uv[1] = Point2f(1, 0);
            uv[2] = Point2f(1, 1);
        }

        Vector2f duv02 = uv[0] - uv[2], duv12 = uv[1] - uv[2];
        Vector3f dp02 = p0 - p2;
        Vector3f dp12 = p1 - p2;
        FloatType determinant = difference_of_products(duv02[0], duv12[1], duv02[1], duv12[0]);

        Vector3f dpdu, dpdv;
        bool degenerateUV = std::abs(determinant) < 1e-9f;
        if (!degenerateUV) {
            // Compute triangle $\dpdu$ and $\dpdv$ via matrix inversion
            FloatType invdet = 1 / determinant;
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
            printf("\nTriangle::interaction_from_intersection():\n");
            printf("    mesh->n and mesh->s are not implemented\n\n");
            asm("trap;");
        }

        return isect;
    }
};

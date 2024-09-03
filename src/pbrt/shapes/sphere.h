#pragma once

#include "pbrt/base/interaction.h"
#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/euclidean_space/transform.h"
#include "pbrt/util/macro.h"
#include <vector>

class ShapeSample;
class ShapeSampleContext;
class ParameterDictionary;

class Sphere {
  public:
    static const Sphere *create(const Transform &render_from_object,
                                const Transform &object_from_render, bool reverse_orientation,
                                const ParameterDictionary &parameters,
                                std::vector<void *> &gpu_dynamic_pointers);

    void init(const Transform &_render_from_object, const Transform &_object_from_render,
              bool _reverse_orientation, FloatType _radius, FloatType _z_min, FloatType _z_max,
              FloatType _phi_max);

    PBRT_CPU_GPU
    Bounds3f bounds() const;

    PBRT_CPU_GPU
    FloatType area() const {
        return phi_max * radius * (z_max - z_min);
    }

    PBRT_GPU
    bool fast_intersect(const Ray &r, FloatType t_max) const {
        return basic_intersect(r, t_max).has_value();
    }

    PBRT_GPU
    cuda::std::optional<ShapeIntersection> intersect(const Ray &ray,
                                                     FloatType t_max = Infinity) const {
        auto isect = basic_intersect(ray, t_max);
        if (!isect) {
            return {};
        }

        SurfaceInteraction intr = interaction_from_intersection(*isect, -ray.d);
        return ShapeIntersection{intr, isect->t_hit};
    }

    PBRT_GPU
    cuda::std::optional<ShapeSample> sample(const Point2f &u) const;

    PBRT_GPU
    cuda::std::optional<ShapeSample> sample(const ShapeSampleContext &ctx, const Point2f &u) const;

    PBRT_GPU
    FloatType pdf(const ShapeSampleContext &ctx, const Vector3f &wi) const;

  private:
    FloatType radius;
    FloatType z_min;
    FloatType z_max;

    FloatType theta_z_min;
    FloatType theta_z_max;
    FloatType phi_max;

    Transform render_from_object;
    Transform object_from_render;

    bool reverse_orientation;
    bool transform_swaps_handedness;

    PBRT_GPU
    cuda::std::optional<QuadricIntersection> basic_intersect(const Ray &r, FloatType tMax) const;

    PBRT_CPU_GPU
    SurfaceInteraction interaction_from_intersection(const QuadricIntersection &isect,
                                                     Vector3f wo) const {
        Point3f pHit = isect.p_obj;
        FloatType phi = isect.phi;
        // Find parametric representation of sphere hit
        FloatType u = phi / phi_max;
        FloatType cosTheta = pHit.z / radius;
        FloatType theta = safe_acos(cosTheta);

        FloatType v = (theta - theta_z_min) / (theta_z_max - theta_z_min);
        // Compute sphere $\dpdu$ and $\dpdv$
        FloatType zRadius = std::sqrt(sqr(pHit.x) + sqr(pHit.y));
        FloatType cosPhi = pHit.x / zRadius, sinPhi = pHit.y / zRadius;
        Vector3f dpdu(-phi_max * pHit.y, phi_max * pHit.x, 0);

        FloatType sinTheta = safe_sqrt(1 - sqr(cosTheta));
        Vector3f dpdv = (theta_z_max - theta_z_min) *
                        Vector3f(pHit.z * cosPhi, pHit.z * sinPhi, -radius * sinTheta);

        // Compute sphere $\dndu$ and $\dndv$
        Vector3f d2Pduu = -phi_max * phi_max * Vector3f(pHit.x, pHit.y, 0);
        Vector3f d2Pduv =
            (theta_z_max - theta_z_min) * pHit.z * phi_max * Vector3f(-sinPhi, cosPhi, 0.);
        Vector3f d2Pdvv = -sqr(theta_z_max - theta_z_min) * Vector3f(pHit.x, pHit.y, pHit.z);

        // Compute coefficients for fundamental forms
        FloatType E = dpdu.dot(dpdu);
        FloatType F = dpdu.dot(dpdv);
        FloatType G = dpdv.dot(dpdv);

        Vector3f n = dpdu.cross(dpdv).normalize();

        FloatType e = n.dot(d2Pduu);
        FloatType f = n.dot(d2Pduv);
        FloatType g = n.dot(d2Pdvv);

        // Compute $\dndu$ and $\dndv$ from fundamental form coefficients
        FloatType EGF2 = difference_of_products(E, G, F, F);

        FloatType invEGF2 = (EGF2 == 0) ? FloatType(0) : 1 / EGF2;
        Normal3f dndu =
            Normal3f((f * F - e * G) * invEGF2 * dpdu + (e * F - f * E) * invEGF2 * dpdv);
        Normal3f dndv =
            Normal3f((g * F - f * G) * invEGF2 * dpdu + (f * F - g * E) * invEGF2 * dpdv);

        // Compute error bounds for sphere intersection
        Vector3f pError = gamma(5) * pHit.to_vector3().abs();

        // Return _SurfaceInteraction_ for quadric intersection
        bool flipNormal = reverse_orientation ^ transform_swaps_handedness;
        Vector3f woObject = object_from_render(wo);

        return render_from_object(SurfaceInteraction(Point3fi(pHit, pError), Point2f(u, v),
                                                     woObject, dpdu, dpdv, dndu, dndv, flipNormal));
    }
};

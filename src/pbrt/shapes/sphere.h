#pragma once

#include <pbrt/base/interaction.h>
#include <pbrt/euclidean_space/bounds3.h>
#include <pbrt/euclidean_space/transform.h>
#include <pbrt/gpu/macro.h>

class GPUMemoryAllocator;
class ParameterDictionary;

struct ShapeSample;
struct ShapeSampleContext;

class Sphere {
  public:
    Sphere(const Transform &_render_from_object, const Transform &_object_from_render,
           bool _reverse_orientation, const ParameterDictionary &parameters);

    PBRT_CPU_GPU
    Bounds3f bounds() const;

    PBRT_CPU_GPU
    Real area() const {
        return phi_max * radius * (z_max - z_min);
    }

    PBRT_CPU_GPU
    bool fast_intersect(const Ray &r, const Real t_max) const {
        return basic_intersect(r, t_max).has_value();
    }

    PBRT_CPU_GPU
    pbrt::optional<ShapeIntersection> intersect(const Ray &ray, Real t_max = Infinity) const {
        auto isect = basic_intersect(ray, t_max);
        if (!isect) {
            return {};
        }

        SurfaceInteraction surface_interaction = interaction_from_intersection(*isect, -ray.d);
        return ShapeIntersection{surface_interaction, isect->t_hit};
    }

    PBRT_CPU_GPU
    pbrt::optional<ShapeSample> sample(const Point2f &u) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeSample> sample(const ShapeSampleContext &ctx, const Point2f &u) const;

    PBRT_CPU_GPU
    Real pdf(const ShapeSampleContext &ctx, const Vector3f &wi) const;

  private:
    const Real radius = NAN;
    Real z_min = NAN;
    Real z_max = NAN;

    Real theta_z_min = NAN;
    Real theta_z_max = NAN;
    Real phi_max = NAN;

    Transform render_from_object;
    Transform object_from_render;

    const bool reverse_orientation = false;
    const bool transform_swaps_handedness = false;

    PBRT_CPU_GPU
    pbrt::optional<QuadricIntersection> basic_intersect(const Ray &r, Real tMax) const;

    PBRT_CPU_GPU
    SurfaceInteraction interaction_from_intersection(const QuadricIntersection &isect,
                                                     Vector3f wo) const {
        Point3f pHit = isect.p_obj;
        Real phi = isect.phi;
        // Find parametric representation of sphere hit
        Real u = phi / phi_max;
        Real cosTheta = pHit.z / radius;
        Real theta = safe_acos(cosTheta);

        Real v = (theta - theta_z_min) / (theta_z_max - theta_z_min);
        // Compute sphere $\dpdu$ and $\dpdv$
        Real zRadius = std::sqrt(sqr(pHit.x) + sqr(pHit.y));
        Real cosPhi = pHit.x / zRadius, sinPhi = pHit.y / zRadius;
        Vector3f dpdu(-phi_max * pHit.y, phi_max * pHit.x, 0);

        Real sinTheta = safe_sqrt(1 - sqr(cosTheta));
        Vector3f dpdv = (theta_z_max - theta_z_min) *
                        Vector3f(pHit.z * cosPhi, pHit.z * sinPhi, -radius * sinTheta);

        // Compute sphere $\dndu$ and $\dndv$
        Vector3f d2Pduu = -phi_max * phi_max * Vector3f(pHit.x, pHit.y, 0);
        Vector3f d2Pduv =
            (theta_z_max - theta_z_min) * pHit.z * phi_max * Vector3f(-sinPhi, cosPhi, 0.);
        Vector3f d2Pdvv = -sqr(theta_z_max - theta_z_min) * Vector3f(pHit.x, pHit.y, pHit.z);

        // Compute coefficients for fundamental forms
        Real E = dpdu.dot(dpdu);
        Real F = dpdu.dot(dpdv);
        Real G = dpdv.dot(dpdv);

        Vector3f n = dpdu.cross(dpdv).normalize();

        Real e = n.dot(d2Pduu);
        Real f = n.dot(d2Pduv);
        Real g = n.dot(d2Pdvv);

        // Compute $\dndu$ and $\dndv$ from fundamental form coefficients
        Real EGF2 = difference_of_products(E, G, F, F);

        Real invEGF2 = (EGF2 == 0) ? Real(0) : 1 / EGF2;
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

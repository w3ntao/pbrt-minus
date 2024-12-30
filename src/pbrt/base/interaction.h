#pragma once

#include "material.h"
#include "pbrt/base/bsdf.h"
#include "pbrt/base/ray.h"
#include "pbrt/euclidean_space/normal3f.h"
#include "pbrt/euclidean_space/point2.h"
#include "pbrt/euclidean_space/point3fi.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"

#include <pbrt/bxdfs/full_bxdf.h>

class BSDF;
class Camera;
class Light;
class Material;
class Sampler;
class SampledWavelengths;

class CoatedConductorBxDF;
class CoatedDiffuseBxDF;
class ConductorBxDF;

class Interaction {
  public:
    Point3fi pi;
    Vector3f wo;
    Normal3f n;
    Point2f uv;

    PBRT_CPU_GPU
    Interaction()
        : pi(Point3fi(NAN, NAN, NAN)), wo(Vector3f(NAN, NAN, NAN)), n(Normal3f(0, 0, 0)),
          uv(Point2f(NAN, NAN)) {}

    PBRT_CPU_GPU
    explicit Interaction(const Point3fi &_pi, const Normal3f &_n, const Point2f &_uv,
                         const Vector3f &_wo)
        : pi(_pi), n(_n), uv(_uv), wo(_wo.normalize()) {}

    PBRT_CPU_GPU
    explicit Interaction(const Point3fi &_pi, const Normal3f &_n, const Point2f &_uv)
        : pi(_pi), n(_n), uv(_uv), wo(Vector3f(NAN, NAN, NAN)) {}

    PBRT_CPU_GPU
    Interaction(const Point3f &_p)
        : pi(_p), n(Normal3f(0, 0, 0)), uv(Point2f(NAN, NAN)), wo(Vector3f(NAN, NAN, NAN)) {}

    PBRT_CPU_GPU
    Interaction(const Point3f &_p, const Normal3f &_n)
        : pi(_p), n(_n), uv(Point2f(NAN, NAN)), wo(Vector3f(NAN, NAN, NAN)) {}

    PBRT_CPU_GPU
    Point3f p() const {
        return pi.to_point3f();
    }

    PBRT_CPU_GPU
    bool is_surface_interaction() const {
        return !n.has_nan() && n != Normal3f(0, 0, 0);
    }

    PBRT_CPU_GPU
    Point3f offset_ray_origin(const Vector3f &w) const {
        return Ray::offset_ray_origin(pi, n, w);
    }

    PBRT_GPU
    Ray spawn_ray(const Vector3f &d) const {
        return Ray(offset_ray_origin(d), d);
    }

    PBRT_CPU_GPU
    Ray spawn_ray_to(const Interaction &it) const {
        return Ray::spawn_ray_to(pi, n, it.pi, it.n);
    }
};

class SurfaceInteraction : public Interaction {
  public:
    Vector3f dpdu, dpdv;
    Normal3f dndu, dndv;
    struct {
        Normal3f n;
        Vector3f dpdu, dpdv;
        Normal3f dndu, dndv;
    } shading;
    int faceIndex = 0;

    Vector3f dpdx;
    Vector3f dpdy;
    FloatType dudx = NAN;
    FloatType dvdx = NAN;
    FloatType dudy = NAN;
    FloatType dvdy = NAN;

    const Material *material;
    const Light *area_light;
    PBRT_CPU_GPU
    SurfaceInteraction()
        : Interaction(Point3fi(NAN, NAN, NAN), Normal3f(0, 0, 0), Point2f(NAN, NAN),
                      Vector3f(NAN, NAN, NAN)),
          dpdu(Vector3f(NAN, NAN, NAN)), dpdv(Vector3f(NAN, NAN, NAN)),
          dndu(Vector3f(NAN, NAN, NAN)), dndv(Vector3f(NAN, NAN, NAN)), material(nullptr),
          area_light(nullptr) {}

    PBRT_CPU_GPU
    explicit SurfaceInteraction(const Point3fi &pi, const Point2f &uv, const Vector3f &wo,
                                const Vector3f &dpdu, const Vector3f &dpdv, Normal3f dndu,
                                const Normal3f &dndv, bool flip_normal)
        : Interaction(pi, Normal3f(dpdu.cross(dpdv).normalize()), uv, wo), dpdu(dpdu), dpdv(dpdv),
          dndu(dndu), dndv(dndv), material(nullptr), area_light(nullptr) {
        // Initialize shading geometry from true geometry
        shading.n = n;
        shading.dpdu = dpdu;
        shading.dpdv = dpdv;
        shading.dndu = dndu;
        shading.dndv = dndv;

        // Adjust normal based on orientation and handedness
        if (flip_normal) {
            n *= -1;
            shading.n *= -1;
        }
    }

    PBRT_GPU
    void compute_differentials(const Ray &ray, const Camera *camera, uint samples_per_pixel);

    PBRT_GPU
    void set_intersection_properties(const Material *_material, const Light *_area_light);

    PBRT_CPU_GPU
    void set_shading_geometry(const Normal3f &ns, const Vector3f &dpdus, const Vector3f &dpdvs,
                              const Normal3f &dndus, const Normal3f &dndvs,
                              bool orientationIsAuthoritative) {
        // Compute _shading.n_ for _SurfaceInteraction_
        shading.n = ns;

        if (orientationIsAuthoritative) {
            n = n.face_forward(shading.n);
        } else {
            shading.n = shading.n.face_forward(n);
        }

        // Initialize _shading_ partial derivative values
        shading.dpdu = dpdus;
        shading.dpdv = dpdvs;
        shading.dndu = dndus;
        shading.dndv = dndvs;

        while (shading.dpdu.squared_length() > 1e16f || shading.dpdv.squared_length() > 1e16f) {
            shading.dpdu /= 1e8f;
            shading.dpdv /= 1e8f;
        }
    }

    PBRT_GPU
    SampledSpectrum le(Vector3f w, const SampledWavelengths &lambda) const;

    PBRT_GPU
    void init_bsdf(BSDF &bsdf, FullBxDF &full_bxdf, const Ray &ray, SampledWavelengths &lambda,
                   const Camera *camera, uint samples_per_pixel);

  private:
    PBRT_GPU
    void init_coated_conductor_bsdf(BSDF &bsdf, CoatedConductorBxDF &coated_conductor_bxdf,
                                    SampledWavelengths &lambda,
                                    const MaterialEvalContext &material_eval_context) const;

    PBRT_GPU
    void init_coated_diffuse_bsdf(BSDF &bsdf, CoatedDiffuseBxDF &coated_diffuse_bxdf,
                                  SampledWavelengths &lambda,
                                  const MaterialEvalContext &material_eval_context) const;

    PBRT_GPU
    void init_conductor_bsdf(BSDF &bsdf, ConductorBxDF &conductor_bxdf, SampledWavelengths &lambda,
                             const MaterialEvalContext &material_eval_context) const;

    PBRT_GPU
    void init_dielectric_bsdf(BSDF &bsdf, DielectricBxDF &dielectric_bxdf,
                              SampledWavelengths &lambda,
                              const MaterialEvalContext &material_eval_context) const;

    PBRT_GPU
    void init_diffuse_bsdf(BSDF &bsdf, DiffuseBxDF &diffuse_bxdf, SampledWavelengths &lambda,
                           const MaterialEvalContext &material_eval_context) const;
};

// ShapeIntersection Definition
struct ShapeIntersection {
    SurfaceInteraction interaction;
    FloatType t_hit;

    PBRT_CPU_GPU ShapeIntersection(const SurfaceInteraction &si, FloatType t)
        : interaction(si), t_hit(t) {}
};

struct QuadricIntersection {
    FloatType t_hit;
    Point3f p_obj;
    FloatType phi;
};

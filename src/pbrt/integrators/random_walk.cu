#include "pbrt/integrators/random_walk.h"

#include "pbrt/base/ray.h"
#include "pbrt/base/sampler.h"
#include "pbrt/base/material.h"

#include "pbrt/accelerator/hlbvh.h"
#include "pbrt/bxdfs/diffuse_bxdf.h"
#include "pbrt/integrators/integrator_base.h"

void RandomWalkIntegrator::init(const IntegratorBase *_base, uint _max_depth) {
    base = _base;
    max_depth = _max_depth;
}

PBRT_GPU
SampledSpectrum RandomWalkIntegrator::li_random_walk(const DifferentialRay &ray,
                                                     SampledWavelengths &lambda, Sampler *sampler,
                                                     uint depth) const {
    auto si = base->bvh->intersect(ray.ray, Infinity);
    if (!si) {
        return SampledSpectrum::same_value(0.0);
    }

    SurfaceInteraction &isect = si->interaction;

    // Get emitted radiance at surface intersection
    Vector3f wo = -ray.ray.d;
    SampledSpectrum radiance_emission = isect.le(wo, lambda);

    // Terminate random walk if maximum depth has been reached
    if (depth >= max_depth) {
        return radiance_emission;
    }

    if (isect.material->get_material_type() == Material::Type::diffuse_material) {
        BSDF bsdf;
        DiffuseBxDF diffuse_bxdf;
        isect.init_diffuse_bsdf(bsdf, diffuse_bxdf, ray, lambda, base->camera, sampler);

        // Randomly sample direction leaving surface for random walk
        Point2f u = sampler->get_2d();
        auto wp = sample_uniform_sphere(u);

        // Evaluate BSDF at surface for sampled direction
        auto fcos = bsdf.f(wo, wp) * wp.abs_dot(isect.shading.n.to_vector3());

        if (!fcos.is_positive()) {
            return radiance_emission;
        }

        auto spawned_ray = isect.spawn_ray(wp);
        return radiance_emission + fcos * li_random_walk(spawned_ray, lambda, sampler, depth + 1) /
                                       (1.0 / (4.0 * compute_pi()));
    }

    REPORT_FATAL_ERROR();
}

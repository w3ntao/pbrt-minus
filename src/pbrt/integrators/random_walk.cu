#include "pbrt/integrators/random_walk.h"
#include "pbrt/base/camera.h"
#include "pbrt/base/sampler.h"
#include "pbrt/base/material.h"
#include "pbrt/bxdfs/diffuse_bxdf.h"

void RandomWalkIntegrator::init(const Camera *_camera, uint _max_depth) {
    camera = _camera;
    max_depth = _max_depth;
}

PBRT_GPU
SampledSpectrum RandomWalkIntegrator::li_random_walk(const Ray &ray, SampledWavelengths &lambda,
                                                     const HLBVH *bvh, Sampler *sampler,
                                                     uint depth) const {
    auto si = bvh->intersect(ray, Infinity);
    if (!si) {
        return SampledSpectrum::same_value(0.0);
    }

    SurfaceInteraction &isect = si->interaction;

    // Get emitted radiance at surface intersection
    Vector3f wo = -ray.d;
    SampledSpectrum radiance_emission = isect.le(wo, lambda);

    // Terminate random walk if maximum depth has been reached
    if (depth >= max_depth) {
        return radiance_emission;
    }

    bool material_is_mix = false;
    if (material_is_mix) {
        // TODO: this part was not handled
    }

    auto fcos = SampledSpectrum::same_value(NAN);
    BSDF bsdf;
    Vector3f wp(NAN, NAN, NAN);

    isect.compute_differentials(ray, camera, 4);
    // TODO: get samples_per_pixel from sampler

    bool material_type_is_diffuse = true;
    if (material_type_is_diffuse) {
        DiffuseBxDF diffuse_bxdf;
        isect.init_diffuse_bsdf(bsdf, diffuse_bxdf, ray, lambda, camera, sampler);

        // Randomly sample direction leaving surface for random walk
        Point2f u = sampler->get_2d();
        wp = sample_uniform_sphere(u);

        // Evaluate BSDF at surface for sampled direction
        fcos = bsdf.f(wo, wp) * abs(wp.dot(isect.shading.n.to_vector3()));
    }

    if (!fcos.is_positive()) {
        return radiance_emission;
    }

    auto spawned_ray = isect.spawn_ray(wp);
    return radiance_emission + fcos * li_random_walk(spawned_ray, lambda, bvh, sampler, depth + 1) /
                                   (1.0 / (4.0 * compute_pi()));
}

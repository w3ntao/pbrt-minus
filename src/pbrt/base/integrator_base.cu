#include <pbrt/accelerator/hlbvh.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/interaction.h>
#include <pbrt/medium/homogeneous_medium.h>

PBRT_CPU_GPU
bool IntegratorBase::fast_intersect(const Ray &ray, Real t_max) const {
    return bvh->fast_intersect(ray, t_max);
}

PBRT_CPU_GPU
bool IntegratorBase::unoccluded(const Interaction &p0, const Interaction &p1) const {
    return !fast_intersect(p0.spawn_ray_to(p1), 0.6) && !fast_intersect(p1.spawn_ray_to(p0), 0.6);
}

PBRT_CPU_GPU
pbrt::optional<ShapeIntersection> IntegratorBase::intersect(const Ray &ray, Real t_max) const {
    return bvh->intersect(ray, t_max);
}

PBRT_CPU_GPU
SampledSpectrum IntegratorBase::compute_transmittance(const Interaction &p0, const Interaction &p1,
                                                      const SampledWavelengths &lambda,
                                                      const int max_depth) const {
    const auto epsilon_distance = p0.p().distance(p1.p()) * ShadowEpsilon;

    auto shadow_ray = p0.spawn_ray_to(p1, true);
    bool possible_self_intersection = false;

    SampledSpectrum transmittance = 1;
    for (auto depth = 0; depth < max_depth * 2; depth++) {
        const auto distance_to_light = p1.p().distance(shadow_ray.o);
        auto optional_intersection =
            this->intersect(shadow_ray, (1.0 - ShadowEpsilon) * distance_to_light);
        const auto next_t =
            optional_intersection ? optional_intersection->t_hit : distance_to_light;

        if (shadow_ray.medium) {
            const SampledSpectrum sigma_t = shadow_ray.medium->sample_sigma_t(lambda);
            transmittance *= SampledSpectrum::exp(-sigma_t * next_t);
        }

        if (!optional_intersection) {
            return transmittance;
        }

        // ray hit something in between light and origin
        if (optional_intersection->interaction.material) {
            // got blocked by some primitives
            return 0;
        }

        // otherwise hit material-less shape

        optional_intersection->interaction.n = Normal3f(shadow_ray.d);
        shadow_ray = optional_intersection->interaction.spawn_ray(shadow_ray.d);

        if (optional_intersection->t_hit < epsilon_distance) {
            if (possible_self_intersection) {
                // forcibly offset shadow_ray.o to avoid self-intersection
                shadow_ray.o += epsilon_distance * shadow_ray.d;
                possible_self_intersection = false;
            } else {
                possible_self_intersection = true;
            }
        }
    }

    // fail to connect 2 points within limited depth
    return 0;
}

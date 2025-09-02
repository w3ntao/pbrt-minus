#include <pbrt/accelerator/hlbvh.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/sampler.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/integrators/megakernel_path.h>
#include <pbrt/light_samplers/power_light_sampler.h>
#include <pbrt/lights/image_infinite_light.h>
#include <pbrt/medium/homogeneous_medium.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectra/densely_sampled_spectrum.h>
#include <pbrt/util/russian_roulette.h>


PBRT_CPU_GPU
SampledSpectrum MegakernelPathIntegrator::evaluate_Li_volume(const Ray &primary_ray,
                                                             SampledWavelengths &lambda,
                                                             const IntegratorBase *base,
                                                             Sampler *sampler, int max_depth,
                                                             bool regularize) {
    auto L = SampledSpectrum(0.0);
    auto beta = SampledSpectrum(1.0);
    bool specular_bounce = false;
    bool any_non_specular_bounces = false;

    pbrt::optional<Real> prev_direction_pdf;
    pbrt::optional<SurfaceInteraction> prev_interaction;
    auto multi_transmittance_pdf = SampledSpectrum(1.0);

    auto next_time_russian_roulette = start_russian_roulette;

    auto ray = primary_ray;
    Real depth = 0.0;
    // making depth float type to counter for contribution from material-less hit
    while (depth < max_depth) {
        if (depth >= next_time_russian_roulette &&
            russian_roulette(beta, sampler, &next_time_russian_roulette)) {
            break;
        }

        auto optional_intersection = base->intersect(ray, Infinity);
        if (ray.medium) {
            const SampledSpectrum sigma_a = ray.medium->sigma_a->sample(lambda);
            const SampledSpectrum sigma_s = ray.medium->sigma_s->sample(lambda);
            const SampledSpectrum sigma_t = sigma_a + sigma_s;

            const auto t_transmittance =
                optional_intersection ? optional_intersection->t_hit : Infinity;
            const auto u = sampler->get_1d();
            if (const auto t = -std::log(1 - u) / sigma_t.average(); t < t_transmittance) {
                ray.o = ray.at(t);
                beta /= sigma_t;

                SurfaceInteraction medium_interaction;
                medium_interaction.pi = ray.o;
                medium_interaction.wo = -ray.d;
                medium_interaction.medium = ray.medium;

                SampledSpectrum Ld =
                    sample_Ld_volume(medium_interaction, nullptr, lambda, base, sampler, max_depth);
                L += beta * Ld * sigma_s;

                auto phase_sample = ray.medium->phase.sample(-ray.d, sampler->get_2d());
                if (!phase_sample) {
                    break;
                }

                beta *= phase_sample->rho / phase_sample->pdf * sigma_s;

                prev_direction_pdf = phase_sample->pdf;

                prev_interaction = medium_interaction;
                multi_transmittance_pdf = SampledSpectrum(1.0);
                ray.d = phase_sample->wi;

                depth += 1;
                continue;
            }

            auto transmittance_pdf = SampledSpectrum::exp(-sigma_t * t_transmittance);
            multi_transmittance_pdf *= transmittance_pdf;
        }

        if (!optional_intersection && beta.is_positive()) {
            // Incorporate emission from infinite lights for escaped ray
            for (int idx = 0; idx < base->infinite_light_num; ++idx) {
                auto light = base->infinite_lights[idx];
                auto Le = light->le(ray, lambda);

                if (depth == 0 || specular_bounce) {
                    L += beta * Le;
                } else {
                    // Compute MIS weight for infinite light
                    Real pdf_light = base->light_sampler->pmf(light) *
                                     light->pdf_li(*prev_interaction, ray.d, true);
                    Real dir_pdf = *prev_direction_pdf * multi_transmittance_pdf.average();

                    Real w = power_heuristic(1, dir_pdf, 1, pdf_light);

                    L += beta * w * Le;
                }
            }

            break;
        }

        SurfaceInteraction &surface_interaction = optional_intersection->interaction;
        if (!surface_interaction.material) {
            // pass through material-less interface
            ray = surface_interaction.spawn_ray(ray.d);
            depth += interface_bounce_contribution;
            continue;
        }

        // Incorporate emission from surface hit by ray
        if (const SampledSpectrum Le = surface_interaction.le(-ray.d, lambda); Le.is_positive()) {
            if (depth == 0 || specular_bounce) {
                L += beta * Le;
            } else {
                // Compute MIS weight for area light
                auto area_light = surface_interaction.area_light;

                Real pdf_light = base->light_sampler->pmf(area_light) *
                                 area_light->pdf_li(*prev_interaction, ray.d);

                Real dir_pdf = *prev_direction_pdf * multi_transmittance_pdf.average();

                Real w = power_heuristic(1, dir_pdf, 1, pdf_light);

                L += beta * w * Le;
            }
        }

        auto bsdf =
            surface_interaction.get_bsdf(lambda, base->camera, sampler->get_samples_per_pixel());
        if (regularize && any_non_specular_bounces) {
            bsdf.regularize();
        }

        // Sample direct illumination from the light sources

        if (pbrt::is_non_specular(bsdf.flags())) {
            SampledSpectrum Ld =
                sample_Ld_volume(surface_interaction, &bsdf, lambda, base, sampler, max_depth);
            L += beta * Ld;
        }

        Vector3f wo = -ray.d;
        auto bs = bsdf.sample_f(wo, sampler->get_1d(), sampler->get_2d());
        if (!bs) {
            break;
        }

        beta *= bs->f * bs->wi.abs_dot(surface_interaction.shading.n.to_vector3()) / bs->pdf;

        prev_direction_pdf = bs->pdf_is_proportional ? bsdf.pdf(wo, bs->wi) : bs->pdf;
        multi_transmittance_pdf = SampledSpectrum(1.0);

        specular_bounce = bs->is_specular();
        any_non_specular_bounces |= !specular_bounce;

        prev_interaction = surface_interaction;
        ray = surface_interaction.spawn_ray(bs->wi);
        // different with PBRT-v4: ignore the DifferentialRay

        depth += 1;
    }

    return L;
}

PBRT_CPU_GPU
SampledSpectrum MegakernelPathIntegrator::li(const Ray &primary_ray, SampledWavelengths &lambda,
                                             Sampler *sampler) const {
    return evaluate_Li_volume(primary_ray, lambda, base, sampler, max_depth, regularize);
}

PBRT_CPU_GPU
SampledSpectrum
MegakernelPathIntegrator::sample_Ld_volume(const SurfaceInteraction &surface_interaction,
                                           const BSDF *bsdf, const SampledWavelengths &lambda,
                                           const IntegratorBase *base, Sampler *sampler,
                                           const int max_depth) {
    LightSampleContext ctx(surface_interaction);
    // Try to nudge the light sampling position to correct side of the surface
    if (bsdf) {
        const BxDFFlags flags = bsdf->flags();
        if (pbrt::is_reflective(flags) && !pbrt::is_transmissive(flags)) {
            ctx.pi = surface_interaction.offset_ray_origin(surface_interaction.wo);
        } else if (pbrt::is_transmissive(flags) && !pbrt::is_reflective(flags)) {
            ctx.pi = surface_interaction.offset_ray_origin(-surface_interaction.wo);
        }
    }

    // Choose a light source for the direct lighting calculation
    const Real u = sampler->get_1d();
    const Point2f u_light = sampler->get_2d();

    auto sampled_light = base->light_sampler->sample(ctx, u);
    if (!sampled_light) {
        return SampledSpectrum(0);
    }

    // Sample a point on the light source for direct lighting
    const auto light = sampled_light->light;
    const auto ls = light->sample_li(ctx, u_light, lambda);
    if (!ls || !ls->l.is_positive() || ls->pdf == 0) {
        return SampledSpectrum(0);
    }

    auto T_light = SampledSpectrum(1.0);
    auto pdf_transmittance_dir = SampledSpectrum(1.0); // for multiple importance sampling

    const auto epsilon_distance = surface_interaction.p().distance(ls->p_light.p()) * ShadowEpsilon;

    auto shadow_ray = surface_interaction.spawn_ray_to(ls->p_light, true);
    bool reach_light = false;
    bool possible_self_intersection = false;
    for (auto depth = 0; depth < max_depth * 2; depth++) {
        const auto distance_to_light = ls->p_light.p().distance(shadow_ray.o);
        auto optional_intersection =
            base->intersect(shadow_ray, (1.0 - ShadowEpsilon) * distance_to_light);
        const auto next_t =
            optional_intersection ? optional_intersection->t_hit : distance_to_light;

        if (shadow_ray.medium) {
            const SampledSpectrum sigma_t = shadow_ray.medium->sample_sigma_t(lambda);
            const auto transmittance = SampledSpectrum::exp(-sigma_t * next_t);

            T_light *= transmittance;
            pdf_transmittance_dir *= transmittance;
        }

        if (!optional_intersection) {
            reach_light = true;
            break;
        }

        // ray hit something in between light and origin
        if (optional_intersection->interaction.material) {
            // got blocked by some primitives
            return SampledSpectrum(0);
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

    if (!reach_light) {
        return SampledSpectrum(0);
    }

    const auto pdf_light = sampled_light->pdf * ls->pdf;
    const auto light_contribution = ls->l / pdf_light * T_light;

    if (bsdf) {
        // Evaluate BSDF for light sample and check light visibility
        Vector3f wo = surface_interaction.wo;
        Vector3f wi = ls->wi;
        SampledSpectrum f =
            bsdf->f(wo, wi) * wi.abs_dot(surface_interaction.shading.n.to_vector3());

        if (!f.is_positive() || !base->unoccluded(surface_interaction, ls->p_light)) {
            return SampledSpectrum(0);
        }

        // Return light's contribution to reflected radiance
        if (pbrt::is_delta_light(light->get_light_type())) {
            return f * light_contribution;
        }

        // for non delta light
        const auto pdf_bsdf = bsdf->pdf(wo, wi) * pdf_transmittance_dir.average();
        const auto w = power_heuristic(1, pdf_light, 1, pdf_bsdf);

        return w * f * light_contribution;
    }
    // else: sample Ld in volume

    auto phase_sample =
        surface_interaction.medium->phase.sample(surface_interaction.wo, sampler->get_2d());

    if (!phase_sample) {
        return SampledSpectrum(0);
    }

    if (pbrt::is_delta_light(light->get_light_type())) {
        // NO MIS for delta light
        return phase_sample->rho * light_contribution;
    }

    const auto pdf_phase = phase_sample->pdf * pdf_transmittance_dir.average();
    const auto w = power_heuristic(1, pdf_light, 1, pdf_phase);

    return w * phase_sample->rho * light_contribution;
}

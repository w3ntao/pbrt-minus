#include <pbrt/accelerator/hlbvh.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/sampler.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/integrators/megakernel_path.h>
#include <pbrt/light_samplers/power_light_sampler.h>
#include <pbrt/lights/image_infinite_light.h>
#include <pbrt/scene/parameter_dictionary.h>

const MegakernelPathIntegrator *
MegakernelPathIntegrator::create(const ParameterDictionary &parameters,
                                 const IntegratorBase *integrator_base,
                                 GPUMemoryAllocator &allocator) {
    auto path_integrator = allocator.allocate<MegakernelPathIntegrator>();

    auto max_depth = parameters.get_integer("maxdepth", 5);
    auto regularize = parameters.get_bool("regularize", false);

    path_integrator->init(integrator_base, max_depth, regularize);

    return path_integrator;
}

void MegakernelPathIntegrator::init(const IntegratorBase *_base, int _max_depth, bool _regularize) {
    base = _base;
    max_depth = _max_depth;
    regularize = _regularize;
}

PBRT_CPU_GPU
SampledSpectrum MegakernelPathIntegrator::evaluate_li(const Ray &primary_ray,
                                                      SampledWavelengths &lambda,
                                                      const IntegratorBase *base, Sampler *sampler,
                                                      int max_depth, bool regularize) {
    auto L = SampledSpectrum(0.0);
    auto beta = SampledSpectrum(1.0);
    bool specular_bounce = false;
    bool any_non_specular_bounces = false;

    int depth = 0;
    pbrt::optional<Real> pdf_bsdf;
    pbrt::optional<LightSampleContext> prev_interaction_light_sample_ctx;

    auto ray = primary_ray;

    // Sample path from camera and accumulate radiance estimate
    while (true) {
        // Trace ray and find closest path vertex and its BSDF
        auto si = base->intersect(ray, Infinity);
        // Add emitted light at intersection point or from the environment
        if (!si) {
            // Incorporate emission from infinite lights for escaped ray
            for (int idx = 0; idx < base->infinite_light_num; ++idx) {
                auto light = base->infinite_lights[idx];
                auto Le = light->le(ray, lambda);

                if (depth == 0 || specular_bounce) {
                    L += beta * Le;
                } else {
                    // Compute MIS weight for infinite light
                    Real pdf_light =
                        base->light_sampler->pmf(*prev_interaction_light_sample_ctx, light) *
                        light->pdf_li(*prev_interaction_light_sample_ctx, ray.d, true);
                    Real weight_bsdf = power_heuristic(1, *pdf_bsdf, 1, pdf_light);

                    L += beta * weight_bsdf * Le;
                }
            }

            break;
        }

        SurfaceInteraction &isect = si->interaction;

        // Incorporate emission from surface hit by ray
        SampledSpectrum Le = isect.le(-ray.d, lambda);
        if (Le.is_positive()) {
            if (depth == 0 || specular_bounce) {
                L += beta * Le;
            } else {
                // Compute MIS weight for area light
                auto area_light = isect.area_light;

                Real pdf_light =
                    base->light_sampler->pmf(*prev_interaction_light_sample_ctx, area_light) *
                    area_light->pdf_li(*prev_interaction_light_sample_ctx, ray.d);
                Real weight_light = power_heuristic(1, *pdf_bsdf, 1, pdf_light);

                L += beta * weight_light * Le;
            }
        }

        auto bsdf = isect.get_bsdf(lambda, base->camera, sampler->get_samples_per_pixel());

        if (regularize && any_non_specular_bounces) {
            bsdf.regularize();
        }

        if (depth++ == max_depth) {
            break;
        }

        // Sample direct illumination from the light sources

        if (pbrt::is_non_specular(bsdf.flags())) {
            SampledSpectrum Ld = sample_ld(isect, &bsdf, lambda, base, sampler);
            L += beta * Ld;
        }

        // Sample BSDF to get new path direction
        Vector3f wo = -ray.d;
        Real u = sampler->get_1d();
        auto bs = bsdf.sample_f(wo, u, sampler->get_2d());
        if (!bs) {
            break;
        }

        // Update path state variables after surface scattering

        beta *= bs->f * bs->wi.abs_dot(isect.shading.n.to_vector3()) / bs->pdf;
        pdf_bsdf = bs->pdf_is_proportional ? bsdf.pdf(wo, bs->wi) : bs->pdf;

        specular_bounce = bs->is_specular();
        any_non_specular_bounces |= !bs->is_specular();

        prev_interaction_light_sample_ctx = isect;
        ray = isect.spawn_ray(bs->wi);
        // different with PBRT-v4: ignore the DifferentialRay

        // Possibly terminate the path with Russian roulette
        if (depth > 8) {
            // depth-8 and clamped-to-0.95 are taken from Mitsuba
            if (beta.max_component_value() < 1) {
                auto q = clamp<Real>(1 - beta.max_component_value(), 0, 0.95);

                if (sampler->get_1d() < q) {
                    break;
                }
                beta /= 1 - q;
            }
        }
    }

    return L;
}

PBRT_CPU_GPU
SampledSpectrum MegakernelPathIntegrator::li(const Ray &primary_ray, SampledWavelengths &lambda,
                                             Sampler *sampler) const {
    return evaluate_li(primary_ray, lambda, base, sampler, max_depth, regularize);
}

PBRT_CPU_GPU
SampledSpectrum MegakernelPathIntegrator::sample_ld(const SurfaceInteraction &intr,
                                                    const BSDF *bsdf, SampledWavelengths &lambda,
                                                    const IntegratorBase *base, Sampler *sampler) {
    // Initialize _LightSampleContext_ for light sampling
    LightSampleContext ctx(intr);
    // Try to nudge the light sampling position to correct side of the surface
    BxDFFlags flags = bsdf->flags();
    if (pbrt::is_reflective(flags) && !pbrt::is_transmissive(flags)) {
        ctx.pi = intr.offset_ray_origin(intr.wo);
    } else if (pbrt::is_transmissive(flags) && !pbrt::is_reflective(flags)) {
        ctx.pi = intr.offset_ray_origin(-intr.wo);
    }

    // Choose a light source for the direct lighting calculation
    Real u = sampler->get_1d();
    auto sampled_light = base->light_sampler->sample(ctx, u);

    Point2f uLight = sampler->get_2d();
    if (!sampled_light) {
        return SampledSpectrum(0);
    }

    // Sample a point on the light source for direct lighting
    auto light = sampled_light->light;
    auto ls = light->sample_li(ctx, uLight, lambda);
    if (!ls || !ls->l.is_positive() || ls->pdf == 0) {
        return SampledSpectrum(0);
    }

    // Evaluate BSDF for light sample and check light visibility
    Vector3f wo = intr.wo;
    Vector3f wi = ls->wi;
    SampledSpectrum f = bsdf->f(wo, wi) * wi.abs_dot(intr.shading.n.to_vector3());

    if (!f.is_positive() || !base->unoccluded(intr, ls->p_light)) {
        return SampledSpectrum(0);
    }

    // Return light's contribution to reflected radiance
    Real pdf_light = sampled_light->pdf * ls->pdf;
    if (pbrt::is_delta_light(light->get_light_type())) {
        return ls->l * f / pdf_light;
    }

    // for non delta light
    Real pdf_bsdf = bsdf->pdf(wo, wi);
    Real weight_light = power_heuristic(1, pdf_light, 1, pdf_bsdf);

    return weight_light * ls->l * f / pdf_light;
}

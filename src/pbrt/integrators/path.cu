#include "pbrt/accelerator/hlbvh.h"
#include "pbrt/base/bxdf.h"
#include "pbrt/base/integrator_base.h"
#include "pbrt/base/interaction.h"
#include "pbrt/base/material.h"
#include "pbrt/base/sampler.h"
#include "pbrt/bxdfs/coated_conductor_bxdf.h"
#include "pbrt/bxdfs/coated_diffuse_bxdf.h"
#include "pbrt/bxdfs/conductor_bxdf.h"
#include "pbrt/bxdfs/dielectric_bxdf.h"
#include "pbrt/bxdfs/diffuse_bxdf.h"
#include "pbrt/integrators/path.h"
#include "pbrt/lights/diffuse_area_light.h"
#include "pbrt/lights/image_infinite_light.h"
#include "pbrt/scene/parameter_dictionary.h"

const PathIntegrator *PathIntegrator::create(const ParameterDictionary &parameters,
                                             const IntegratorBase *integrator_base,
                                             std::vector<void *> &gpu_dynamic_pointers) {
    PathIntegrator *path_integrator;
    CHECK_CUDA_ERROR(cudaMallocManaged(&path_integrator, sizeof(PathIntegrator)));
    gpu_dynamic_pointers.push_back(path_integrator);

    auto max_depth = parameters.get_integer("maxdepth", 5);

    path_integrator->init(integrator_base, max_depth);

    return path_integrator;
}

void PathIntegrator::init(const IntegratorBase *_base, uint _max_depth) {
    base = _base;
    max_depth = _max_depth;
    regularize = true;
}

PBRT_GPU SampledSpectrum PathIntegrator::li(const Ray &primary_ray, SampledWavelengths &lambda,
                                            Sampler *sampler) {
    auto L = SampledSpectrum(0.0);
    auto beta = SampledSpectrum(1.0);
    bool specular_bounce = true;
    bool any_non_specular_bounces = false;

    uint path_length = 0;
    FloatType pdf_bsdf = NAN;
    FloatType eta_scale = 1.0;
    LightSampleContext prev_interaction_light_sample_ctx;

    auto ray = primary_ray;

    BSDF bsdf;
    CoatedConductorBxDF coated_conductor_bxdf;
    CoatedDiffuseBxDF coated_diffuse_bxdf;
    ConductorBxDF conductor_bxdf;
    DielectricBxDF dielectric_bxdf;
    DiffuseBxDF diffuse_bxdf;

    // Sample path from camera and accumulate radiance estimate
    while (true) {
        // Trace ray and find closest path vertex and its BSDF
        auto si = base->bvh->intersect(ray, Infinity);
        // Add emitted light at intersection point or from the environment
        if (!si) {
            // Incorporate emission from infinite lights for escaped ray
            for (uint idx = 0; idx < base->infinite_light_num; ++idx) {
                auto light = base->infinite_lights[idx];
                auto Le = light->le(ray, lambda);

                if (path_length == 0 || specular_bounce) {
                    L += beta * Le;
                } else {
                    // Compute MIS weight for infinite light
                    FloatType pdf_light =
                        base->light_sampler->pmf(prev_interaction_light_sample_ctx, light) *
                        light->pdf_li(prev_interaction_light_sample_ctx, ray.d, true);
                    FloatType weight_bsdf = power_heuristic(1, pdf_bsdf, 1, pdf_light);

                    L += beta * weight_bsdf * Le;
                }
            }

            break;
        }

        SurfaceInteraction &isect = si->interaction;

        // Incorporate emission from surface hit by ray
        SampledSpectrum Le = isect.le(-ray.d, lambda);
        if (Le.is_positive()) {
            if (path_length == 0 || specular_bounce)
                L += beta * Le;
            else {
                // Compute MIS weight for area light
                auto area_light = isect.area_light;

                FloatType pdf_light =
                    base->light_sampler->pmf(prev_interaction_light_sample_ctx, area_light) *
                    area_light->pdf_li(prev_interaction_light_sample_ctx, ray.d);
                FloatType weight_light = power_heuristic(1, pdf_bsdf, 1, pdf_light);

                L += beta * weight_light * Le;
            }
        }

        switch (isect.material->get_material_type()) {
        case (Material::Type::coated_conductor): {
            isect.init_coated_conductor_bsdf(bsdf, coated_conductor_bxdf, ray, lambda, base->camera,
                                             sampler);
            break;
        }
        case (Material::Type::coated_diffuse): {
            isect.init_coated_diffuse_bsdf(bsdf, coated_diffuse_bxdf, ray, lambda, base->camera,
                                           sampler);
            break;
        }

        case (Material::Type::conductor): {
            isect.init_conductor_bsdf(bsdf, conductor_bxdf, ray, lambda, base->camera, sampler);
            break;
        }

        case (Material::Type::dielectric): {
            isect.init_dielectric_bsdf(bsdf, dielectric_bxdf, ray, lambda, base->camera, sampler);
            break;
        }

        case (Material::Type::diffuse): {
            isect.init_diffuse_bsdf(bsdf, diffuse_bxdf, ray, lambda, base->camera, sampler);
            break;
        }

        case (Material::Type::mix): {
            printf("\nyou should not see MixMaterial here\n\n");
            REPORT_FATAL_ERROR();
        }

        default: {
            REPORT_FATAL_ERROR();
        }
        }

        if (regularize && any_non_specular_bounces) {
            bsdf.regularize();
        }

        if (path_length++ == max_depth) {
            break;
        }

        // Sample direct illumination from the light sources

        if (_is_non_specular(bsdf.flags())) {
            SampledSpectrum Ld = sample_ld(isect, &bsdf, lambda, sampler);
            L += beta * Ld;
        }

        // Sample BSDF to get new path direction
        Vector3f wo = -ray.d;
        FloatType u = sampler->get_1d();
        auto bs = bsdf.sample_f(wo, u, sampler->get_2d());
        if (!bs) {
            break;
        }

        // Update path state variables after surface scattering

        beta *= bs->f * bs->wi.abs_dot(isect.shading.n.to_vector3()) / bs->pdf;
        pdf_bsdf = bs->pdf_is_proportional ? bsdf.pdf(wo, bs->wi) : bs->pdf;

        specular_bounce = bs->is_specular();
        any_non_specular_bounces |= (!bs->is_specular());

        if (bs->is_transmission()) {
            eta_scale *= sqr(bs->eta);
        }
        prev_interaction_light_sample_ctx = isect;
        ray = isect.spawn_ray(bs->wi);
        // different with PBRT-v4: ignore the DifferentialRay

        // Possibly terminate the path with Russian roulette
        if (path_length > 8) {
            SampledSpectrum russian_roulette_beta = beta * eta_scale;
            if (russian_roulette_beta.max_component_value() < 1) {
                auto q = clamp<FloatType>(1 - russian_roulette_beta.max_component_value(), 0, 0.95);

                if (sampler->get_1d() < q) {
                    break;
                }
                beta /= 1 - q;
            }
        }
    }

    return L;
}

PBRT_GPU
SampledSpectrum PathIntegrator::sample_ld(const SurfaceInteraction &intr, const BSDF *bsdf,
                                          SampledWavelengths &lambda, Sampler *sampler) const {
    // Initialize _LightSampleContext_ for light sampling
    LightSampleContext ctx(intr);
    // Try to nudge the light sampling position to correct side of the surface
    BxDFFlags flags = bsdf->flags();
    if (_is_reflective(flags) && !_is_transmissive(flags)) {
        ctx.pi = intr.offset_ray_origin(intr.wo);
    } else if (_is_transmissive(flags) && !_is_reflective(flags)) {
        ctx.pi = intr.offset_ray_origin(-intr.wo);
    }

    // Choose a light source for the direct lighting calculation
    FloatType u = sampler->get_1d();
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
    FloatType pdf_light = sampled_light->p * ls->pdf;
    if (is_deltaLight(light->get_light_type())) {
        return ls->l * f / pdf_light;
    }

    // for non delta light
    FloatType pdf_bsdf = bsdf->pdf(wo, wi);
    FloatType weight_light = power_heuristic(1, pdf_light, 1, pdf_bsdf);

    return weight_light * ls->l * f / pdf_light;
}

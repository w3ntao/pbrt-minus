#include "pbrt/integrators/simple_path.h"

#include "pbrt/accelerator/hlbvh.h"
#include "pbrt/base/bxdf.h"
#include "pbrt/base/sampler.h"
#include "pbrt/base/interaction.h"
#include "pbrt/base/material.h"

#include "pbrt/bxdfs/diffuse_bxdf.h"
#include "pbrt/bxdfs/dielectric_bxdf.h"

#include "pbrt/lights/diffuse_area_light.h"

void SimplePathIntegrator::init(const IntegratorBase *_base, uint _max_depth) {
    base = _base;
    max_depth = _max_depth;
}

PBRT_GPU
bool SimplePathIntegrator::fast_intersect(const Ray &ray, FloatType t_max) const {
    return base->bvh->fast_intersect(ray, t_max);
}

PBRT_GPU SampledSpectrum SimplePathIntegrator::li(const DifferentialRay &primary_ray,
                                                  SampledWavelengths &lambda, Sampler *sampler) {
    auto L = SampledSpectrum::same_value(0.0);
    auto beta = SampledSpectrum::same_value(1.0);
    bool specular_bounce = true;
    uint depth = 0;

    auto ray = primary_ray;

    BSDF bsdf;
    DiffuseBxDF diffuse_bxdf;
    DielectricBxDF dielectric_bxdf;

    while (beta.is_positive()) {
        auto si = base->bvh->intersect(ray.ray, Infinity);
        if (!si) {
            if (specular_bounce) {
                // TODO: sampling infinite lights is not implemented
            }
            break;
        }

        SurfaceInteraction &isect = si->interaction;
        Vector3f wo = -ray.ray.d;

        // Get emitted radiance at surface intersection
        if (specular_bounce) {
            L += beta * isect.le(wo, lambda);
        }

        // End path if maximum depth reached
        depth += 1;
        if (depth == max_depth) {
            break;
        }

        switch (isect.material->get_material_type()) {
        case (Material::Type::diffuse): {
            isect.init_diffuse_bsdf(bsdf, diffuse_bxdf, ray, lambda, base->camera, sampler);
            break;
        }
        case (Material::Type::dieletric): {
            isect.init_dielectric_bsdf(bsdf, dielectric_bxdf, ray, lambda, base->camera, sampler);
            break;
        }
        default: {
            REPORT_FATAL_ERROR();
        }
        }

        // Sample direct illumination if _sampleLights_ is true
        auto sampled_light = base->uniform_light_sampler->sample(sampler->get_1d());

        if (sampled_light.has_value()) {
            auto u_light = sampler->get_2d();
            auto light_li_sample =
                sampled_light->light->sample_li(LightSampleContext(isect), u_light, lambda, false);
            if (light_li_sample.has_value() && light_li_sample->l.is_positive() &&
                light_li_sample->pdf > 0.0) {
                auto wi = light_li_sample->wi;
                SampledSpectrum f = bsdf.f(wo, wi, TransportMode::Radiance) *
                                    wi.abs_dot(isect.shading.n.to_vector3());
                if (f.is_positive() && unoccluded(isect, light_li_sample->p_light)) {
                    L += beta * f * light_li_sample->l / (sampled_light->p * light_li_sample->pdf);
                }
            }
        }

        auto u = sampler->get_1d();
        auto bs = bsdf.sample_f(wo, u, sampler->get_2d());
        if (!bs) {
            break;
        }

        beta *= bs->f * bs->wi.abs_dot(isect.shading.n.to_vector3()) / bs->pdf;

        specular_bounce = bs->is_specular();
        ray = isect.spawn_ray(bs->wi);
    }

    return L;
}

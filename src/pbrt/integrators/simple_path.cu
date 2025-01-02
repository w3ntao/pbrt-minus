#include "pbrt/accelerator/hlbvh.h"
#include "pbrt/base/bxdf.h"
#include "pbrt/base/integrator_base.h"
#include "pbrt/base/interaction.h"
#include "pbrt/base/material.h"
#include "pbrt/base/sampler.h"
#include "pbrt/integrators/simple_path.h"
#include "pbrt/light_samplers/power_light_sampler.h"
#include "pbrt/lights/diffuse_area_light.h"
#include "pbrt/lights/image_infinite_light.h"

const SimplePathIntegrator *
SimplePathIntegrator::create(const ParameterDictionary &parameters,
                             const IntegratorBase *integrator_base,
                             std::vector<void *> &gpu_dynamic_pointers) {
    SimplePathIntegrator *simple_path_integrator;
    CHECK_CUDA_ERROR(cudaMallocManaged(&simple_path_integrator, sizeof(SimplePathIntegrator)));
    gpu_dynamic_pointers.push_back(simple_path_integrator);

    simple_path_integrator->init(integrator_base, 5);

    return simple_path_integrator;
}

void SimplePathIntegrator::init(const IntegratorBase *_base, uint _max_depth) {
    base = _base;
    max_depth = _max_depth;
}

PBRT_GPU SampledSpectrum SimplePathIntegrator::li(const Ray &primary_ray,
                                                  SampledWavelengths &lambda,
                                                  Sampler *sampler) const {
    auto L = SampledSpectrum(0.0);
    auto beta = SampledSpectrum(1.0);
    bool specular_bounce = true;
    uint depth = 0;

    auto ray = primary_ray;

    BSDF bsdf;

    while (beta.is_positive()) {
        auto si = base->intersect(ray, Infinity);

        if (!si) {
            if (specular_bounce) {
                for (uint idx = 0; idx < base->infinite_light_num; ++idx) {
                    auto light = base->infinite_lights[idx];
                    L += beta * light->le(ray, lambda);
                }
            }
            break;
        }

        SurfaceInteraction &isect = si->interaction;
        Vector3f wo = -ray.d;

        // Get emitted radiance at surface intersection
        if (specular_bounce) {
            L += beta * isect.le(wo, lambda);
        }

        // End path if maximum depth reached
        depth += 1;
        if (depth == max_depth) {
            break;
        }

        isect.init_bsdf(bsdf, ray, lambda, base->camera, sampler->get_samples_per_pixel());

        // Sample direct illumination if _sampleLights_ is true
        auto sampled_light = base->light_sampler->sample(sampler->get_1d());

        if (sampled_light.has_value()) {
            auto u_light = sampler->get_2d();
            auto light_li_sample =
                sampled_light->light->sample_li(LightSampleContext(isect), u_light, lambda);

            if (light_li_sample.has_value() && light_li_sample->l.is_positive() &&
                light_li_sample->pdf > 0.0) {
                auto wi = light_li_sample->wi;
                SampledSpectrum f = bsdf.f(wo, wi, TransportMode::Radiance) *
                                    wi.abs_dot(isect.shading.n.to_vector3());

                if (f.is_positive() && base->unoccluded(isect, light_li_sample->p_light)) {
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

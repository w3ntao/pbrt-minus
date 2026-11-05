#include <chrono>
#include <cmath>
#include <cuda/atomic>
#include <pbrt/accelerator/hlbvh.h>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/material.h>
#include <pbrt/base/sampler.h>
#include <pbrt/base/spectrum.h>
#include <pbrt/film/firefly_cascade.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/gui/gl_helper.h>
#include <pbrt/integrators/megakernel_path.h>
#include <pbrt/light_samplers/power_light_sampler.h>
#include <pbrt/lights/image_infinite_light.h>
#include <pbrt/medium/homogeneous_medium.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectra/densely_sampled_spectrum.h>
#include <pbrt/spectrum_util/color_encoding.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/util/russian_roulette.h>
#include <stdexcept>
#include <thread>

using PreviewCounter = cuda::atomic<int, cuda::thread_scope_system>;

__global__ static void megakernel_path_render(Film *film, uint8_t *gpu_frame_buffer, PreviewCounter *counter,
                                              const int samples_per_pixel, Sampler *samplers,
                                              const MegakernelPathIntegrator *integrator, const IntegratorBase *base,
                                              const Spectrum *cie_y, const firefly::CascadeView detector) {
    const auto resolution = film->get_resolution();
    const int pixel_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (pixel_index >= resolution.x * resolution.y) {
        return;
    }

    const int x = pixel_index % resolution.x;
    const int y = pixel_index / resolution.x;
    const Point2i p_pixel(x, y);
    auto sampler = &samplers[pixel_index];

    for (int sample_idx = 0; sample_idx < samples_per_pixel; ++sample_idx) {
        sampler->start_pixel_sample(pixel_index, sample_idx, 0);
        const auto camera_sample = sampler->get_camera_sample(p_pixel, base->filter);
        auto lambda = SampledWavelengths::sample_visible(sampler->get_1d());
        const auto ray = base->camera->generate_ray(camera_sample, sampler);
        const auto radiance_l = ray.weight * integrator->li(ray.ray, lambda, sampler);

        if (DEBUG_MODE && radiance_l.has_nan()) {
            printf("%s(): pixel(%d, %d), samples %d: has NAN\n", __func__, p_pixel.x, p_pixel.y, sample_idx);
        }

        film->add_sample(p_pixel, radiance_l, lambda, camera_sample.filter_weight);

        // Normalized, unfiltered CIE Y, using the same luminance convention as MLT.
        const Real luminance = radiance_l.y(lambda, cie_y);
        detector.record(pixel_index, luminance);
    }

    if (gpu_frame_buffer == nullptr || counter == nullptr) {
        return;
    }

    const auto rgb = film->get_pixel_rgb(p_pixel);
    const SRGBColorEncoding srgb_encoding;
    gpu_frame_buffer[pixel_index * 3 + 0] = rgb.has_nan() ? 0 : srgb_encoding.from_linear(rgb.r);
    gpu_frame_buffer[pixel_index * 3 + 1] = rgb.has_nan() ? 0 : srgb_encoding.from_linear(rgb.g);
    gpu_frame_buffer[pixel_index * 3 + 2] = rgb.has_nan() ? 0 : srgb_encoding.from_linear(rgb.b);

    // Publish completion only after this pixel's preview color has been written.
    counter->fetch_add(1, cuda::memory_order_release);
}

static __global__ void path_denoising_statistics(Film *film, const int samples_per_pixel, Sampler *samplers,
                                                 const IntegratorBase *base, const int max_depth, const bool regularize,
                                                 const bool match_bdpt_depth) {
    const auto resolution = film->get_resolution();
    const int pixel_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (pixel_index >= resolution.x * resolution.y) {
        return;
    }

    const Point2i pixel(pixel_index % resolution.x, pixel_index / resolution.x);
    auto *sampler = &samplers[pixel_index];
    for (int sample_idx = 0; sample_idx < samples_per_pixel; ++sample_idx) {
        sampler->start_pixel_sample(pixel_index, sample_idx, 0);
        const auto camera_sample = sampler->get_camera_sample(pixel, base->filter);
        auto lambda = SampledWavelengths::sample_visible(sampler->get_1d());
        const auto camera_ray = base->camera->generate_ray(camera_sample, sampler);
        const auto radiance = camera_ray.weight * MegakernelPathIntegrator::evaluate_Li_volume(
                                                      nullptr, camera_ray.ray, lambda, sampler, base, max_depth,
                                                      regularize, match_bdpt_depth);
        // RGBFilm records transformed sensor XYZ before returning for weight 0.
        film->add_sample(pixel_index, radiance, lambda, 0);
    }
}

MegakernelPathIntegrator::MegakernelPathIntegrator(const int samples_per_pixel, const Sampler::Type sampler_type,
                                                   const ParameterDictionary &parameters,
                                                   const IntegratorBase *integrator_base, GPUMemoryAllocator &allocator)
    : base(integrator_base), cie_y(parameters.global_spectra->cie_y), samples_per_pixel(samples_per_pixel),
      max_depth(parameters.get_integer("maxdepth", 5)), regularize(parameters.get_bool("regularize", false)) {
    const auto resolution = base->camera->get_camera_base()->resolution;
    samplers = Sampler::create_samplers(sampler_type, samples_per_pixel, resolution.x * resolution.y, allocator);
}

void MegakernelPathIntegrator::render(Film *film, GreyScaleFilm &firefly_mask, const std::string &output_filename,
                                      const bool preview) const {
    const auto resolution = film->get_resolution();
    const auto num_pixels = resolution.x * resolution.y;
    if (samples_per_pixel <= 0) {
        REPORT_FATAL_ERROR();
    }

    GPUMemoryAllocator local_allocator;
    GLHelper gl_helper;
    PreviewCounter *counter = nullptr;
    if (preview) {
        gl_helper.init("initializing", resolution);
        counter = local_allocator.create<PreviewCounter>(0);
    }

    firefly::Cascade cascade(resolution, local_allocator);

    constexpr int threads = 256;
    megakernel_path_render<<<divide_and_ceil(num_pixels, threads), threads>>>(
        film, gl_helper.gpu_frame_buffer, counter, samples_per_pixel, samplers, this, base, cie_y, cascade.view());
    CHECK_CUDA_ERROR(cudaGetLastError());

    if (preview) {
        cudaError_t render_status;
        while ((render_status = cudaStreamQuery(nullptr)) == cudaErrorNotReady) {
            const int completed_pixels = counter->load(cuda::memory_order_acquire);
            gl_helper.draw_frame(GLHelper::assemble_title(double(completed_pixels) / num_pixels));
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        CHECK_CUDA_ERROR(render_status);
    }

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    cascade.finalize(firefly_mask, samples_per_pixel, output_filename);
}

void MegakernelPathIntegrator::render_denoising_statistics(Film *film, const bool match_bdpt_depth) const {
    if (film == nullptr || samples_per_pixel <= 0) {
        throw std::invalid_argument("denoising statistics require a Film and positive samples per pixel");
    }
    const auto resolution = film->get_resolution();
    const int num_pixels = resolution.x * resolution.y;
    constexpr int threads = 256;
    path_denoising_statistics<<<divide_and_ceil(num_pixels, threads), threads>>>(
        film, samples_per_pixel, samplers, base, max_depth, regularize, match_bdpt_depth);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

struct MegakernelPathIntegrator::PathContinuation {
    SampledSpectrum beta;
    bool specular_bounce;
    bool any_non_specular_bounces;
    const SurfaceInteraction *prev_interaction;
    Real prev_direction_pdf;
    int next_time_russian_roulette;
};

PBRT_CPU_GPU
SampledSpectrum MegakernelPathIntegrator::evaluate_Li_volume(Real *depth, const Ray &primary_ray,
                                                             SampledWavelengths &lambda, Sampler *sampler,
                                                             const IntegratorBase *base, int max_depth, bool regularize,
                                                             bool match_bdpt_depth, Real path_depth,
                                                             const PathContinuation *continuation) {
    SampledSpectrum L = 0;
    SampledSpectrum beta = continuation ? continuation->beta : SampledSpectrum(1);
    bool specular_bounce = continuation && continuation->specular_bounce;
    bool any_non_specular_bounces = continuation && continuation->any_non_specular_bounces;

    pbrt::optional<Real> prev_direction_pdf;
    pbrt::optional<SurfaceInteraction> prev_interaction;
    if (continuation) {
        prev_direction_pdf = continuation->prev_direction_pdf;
        prev_interaction = *continuation->prev_interaction;
    }
    SampledSpectrum multi_transmittance_pdf = 1;

    auto next_time_russian_roulette = continuation ? continuation->next_time_russian_roulette : start_russian_roulette;

    auto ray = primary_ray;

    int consecutive_interfaces = 0;
    // Legacy depth includes fractional interface hits. Selective mode counts
    // scattering events and still traces the final segment to an emitter.
    while ((match_bdpt_depth ? path_depth <= max_depth : path_depth < max_depth) && beta.is_positive()) {
        if (path_depth >= next_time_russian_roulette && russian_roulette(beta, sampler, &next_time_russian_roulette)) {
            break;
        }

        auto optional_intersection = base->intersect(ray, Infinity);

        if (ray.medium) {
            const SampledSpectrum sigma_a = ray.medium->sigma_a->sample(lambda);
            const SampledSpectrum sigma_s = ray.medium->sigma_s->sample(lambda);
            const SampledSpectrum sigma_t = sigma_a + sigma_s;

            const auto t_max = optional_intersection ? optional_intersection->t_hit : Infinity;
            if (const auto t = sample_exponential(sampler->get_1d(), sigma_t.average()); t < t_max) {
                if (match_bdpt_depth && path_depth >= max_depth) {
                    // Another medium interaction would exceed the path limit.
                    break;
                }
                // scatter in medium
                ray.o = ray.at(t);
                beta *= sigma_s / sigma_t;

                SurfaceInteraction medium_interaction;
                medium_interaction.pi = ray.o;
                medium_interaction.wo = -ray.d;
                medium_interaction.medium = ray.medium;

                SampledSpectrum Ld = sample_Ld_volume(medium_interaction, nullptr, lambda, sampler, base);
                L += beta * Ld;

                auto phase_sample = ray.medium->phase.sample(-ray.d, sampler->get_2d());
                if (!phase_sample) {
                    break;
                }

                beta *= phase_sample->rho / phase_sample->pdf;

                prev_direction_pdf = phase_sample->pdf;

                prev_interaction = medium_interaction;
                multi_transmittance_pdf = 1;
                ray.d = phase_sample->wi;

                specular_bounce = false;
                any_non_specular_bounces = true;

                path_depth += 1;
                consecutive_interfaces = 0;
                continue;
            }
            // otherwise pass through medium

            multi_transmittance_pdf *= SampledSpectrum::exp(-sigma_t * t_max);
        }

        if (!optional_intersection) {
            // Incorporate emission from infinite lights for escaped ray
            for (int idx = 0; idx < base->infinite_light_num; ++idx) {
                auto light = base->infinite_lights[idx];
                auto Le = light->le(ray, lambda);

                if (path_depth == 0 || specular_bounce) {
                    L += beta * Le;
                } else {
                    // Compute MIS weight for infinite light
                    Real pdf_light = base->light_sampler->pmf(light) * light->pdf_li(*prev_interaction, ray.d, true);
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
            if (match_bdpt_depth) {
                // Interfaces consume no scattering depth, but repeated self-hits
                // still need an independent finite traversal bound.
                if (++consecutive_interfaces >= IntegratorBase::MAX_VOLUME_BOUNCES) {
                    break;
                }
            } else {
                path_depth += IntegratorBase::interface_bounce_contribution;
            }

            continue;
        }

        // Incorporate emission from surface hit by ray
        if (const SampledSpectrum Le = surface_interaction.le(-ray.d, lambda); Le.is_positive()) {
            if (path_depth == 0 || specular_bounce) {
                L += beta * Le;
            } else {
                // Compute MIS weight for area light
                auto area_light = surface_interaction.area_light;

                Real pdf_light = base->light_sampler->pmf(area_light) * area_light->pdf_li(*prev_interaction, ray.d);

                Real dir_pdf = *prev_direction_pdf * multi_transmittance_pdf.average();

                Real w = power_heuristic(1, dir_pdf, 1, pdf_light);

                L += beta * w * Le;
            }

            if (!match_bdpt_depth) {
                break;
            }
        }

        if (match_bdpt_depth && path_depth >= max_depth) {
            // Emission on the terminal segment is included above; scattering
            // and next-event estimation here would add an extra path vertex.
            break;
        }

        auto bsdf = surface_interaction.get_bsdf(lambda, base->camera, sampler->get_samples_per_pixel());
        if (regularize && any_non_specular_bounces) {
            bsdf.regularize();
        }

        // Sample direct illumination from the light sources

        if (pbrt::is_non_specular(bsdf.flags())) {
            SampledSpectrum Ld = sample_Ld_volume(surface_interaction, &bsdf, lambda, sampler, base);
            L += beta * Ld;
        }

        Vector3f wo = -ray.d;
        // ray split:
        // trace both reflection and refraction (rather than sampling one) for the first bounce
        if (path_depth == 0.0 && surface_interaction.material->get_material_type() == Material::Type::dielectric) {
            const auto samples = bsdf.sample_dielectric_f(wo, sampler->get_2d());
            const pbrt::optional<BSDFSample> *branches[] = {&samples.reflection, &samples.transmission};
            Real deepest_path = path_depth;
            // ray split: trace both reflection and transmission branches, if any
            for (const auto *branch : branches) {
                if (!*branch) {
                    continue;
                }
                const auto &sample = **branch;
                const auto flags =
                    sample.is_reflection() ? BxDFReflTransFlags::Reflection : BxDFReflTransFlags::Transmission;
                // Tracing both branches retains Fresnel weights in f; only
                // the conditional direction density divides throughput.
                const Real direction_pdf =
                    sample.is_specular() ? 1 : bsdf.pdf(wo, sample.wi, TransportMode::Radiance, flags);
                if (!(direction_pdf > 0)) {
                    continue;
                }
                const Real cos_theta = sample.wi.abs_dot(surface_interaction.shading.n.to_vector3());
                const PathContinuation child{
                    beta * sample.f * cos_theta / direction_pdf,
                    sample.is_specular(),
                    any_non_specular_bounces || !sample.is_specular(),
                    &surface_interaction,
                    // Preserve the mixture PDF used by direct-light MIS.
                    sample.is_specular() ? sample.pdf : bsdf.pdf(wo, sample.wi),
                    next_time_russian_roulette,
                };
                const auto child_ray = surface_interaction.spawn_ray(sample.wi);
                auto child_lambda = lambda;
                Real child_depth = path_depth + 1;
                const auto child_L = evaluate_Li_volume(&child_depth, child_ray, child_lambda, sampler, base, max_depth,
                                                        regularize, match_bdpt_depth, path_depth + 1, &child);
                // A later dispersive hit may terminate wavelengths on just one
                // branch. Convert its result to the parent's wavelength PDF.
                L += child_L * lambda.pdf_as_sampled_spectrum().safe_div(child_lambda.pdf_as_sampled_spectrum());
                deepest_path = std::max(deepest_path, child_depth);
            }
            path_depth = deepest_path;
            break;
        }

        // othwerwise: not dielectric, sample one direction and continue
        auto bs = bsdf.sample_f(wo, sampler->get_1d(), sampler->get_2d());
        if (!bs) {
            break;
        }

        beta *= bs->f * bs->wi.abs_dot(surface_interaction.shading.n.to_vector3()) / bs->pdf;

        prev_direction_pdf = bs->pdf_is_proportional ? bsdf.pdf(wo, bs->wi) : bs->pdf;
        multi_transmittance_pdf = 1;

        specular_bounce = bs->is_specular();
        any_non_specular_bounces |= !specular_bounce;

        prev_interaction = surface_interaction;
        ray = surface_interaction.spawn_ray(bs->wi);
        // different with PBRT-v4: ignore the DifferentialRay

        path_depth += 1;
        consecutive_interfaces = 0;
    }

    if (depth != nullptr) {
        *depth = path_depth;
    }

    return L;
}

PBRT_CPU_GPU
SampledSpectrum MegakernelPathIntegrator::li(const Ray &primary_ray, SampledWavelengths &lambda,
                                             Sampler *sampler) const {
    return evaluate_Li_volume(nullptr, primary_ray, lambda, sampler, base, max_depth, regularize);
}

PBRT_CPU_GPU
SampledSpectrum MegakernelPathIntegrator::sample_Ld_volume(const SurfaceInteraction &surface_interaction,
                                                           const BSDF *bsdf, const SampledWavelengths &lambda,
                                                           Sampler *sampler, const IntegratorBase *base) {
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
        return 0;
    }

    // Sample a point on the light source for direct lighting
    const auto light = sampled_light->light;
    const auto ls = light->sample_li(ctx, u_light, lambda);
    if (!ls || !ls->l.is_positive() || ls->pdf == 0) {
        return 0;
    }

    const auto transmittance = base->compute_transmittance(surface_interaction, ls->p_light, lambda);

    const auto &pdf_transmittance_dir = transmittance;

    const auto pdf_light = sampled_light->pdf * ls->pdf;
    const auto light_contribution = ls->l / pdf_light * transmittance;

    if (bsdf) {
        // Evaluate BSDF for light sample and check light visibility
        const Vector3f wo = surface_interaction.wo;
        const Vector3f wi = ls->wi;
        const SampledSpectrum f = bsdf->f(wo, wi) * wi.abs_dot(surface_interaction.shading.n.to_vector3());

        if (!f.is_positive() || !base->unoccluded(surface_interaction, ls->p_light)) {
            return 0;
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

    auto phase_sample = surface_interaction.medium->phase.sample(surface_interaction.wo, sampler->get_2d());

    if (!phase_sample) {
        return 0;
    }

    if (pbrt::is_delta_light(light->get_light_type())) {
        // NO MIS for delta light
        return phase_sample->rho * light_contribution;
    }

    const auto pdf_phase = phase_sample->pdf * pdf_transmittance_dir.average();
    const auto w = power_heuristic(1, pdf_light, 1, pdf_phase);

    return w * phase_sample->rho * light_contribution;
}

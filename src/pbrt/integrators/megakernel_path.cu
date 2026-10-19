#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <pbrt/accelerator/hlbvh.h>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/sampler.h>
#include <pbrt/base/spectrum.h>
#include <pbrt/film/firefly_detector.h>
#include <pbrt/film/grey_scale_film.h>
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

static void write_firefly_cascade(Real *const *counts, const Point2i resolution, const std::string &output_filename) {
    const auto image_path = std::filesystem::path(output_filename);
    const int num_pixels = resolution.x * resolution.y;
    for (int layer_index = 0; layer_index < firefly::layer_count; ++layer_index) {
        GreyScaleFilm layer(resolution);
        for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
            // Bitterli's buffers store sample counts, rather than layer radiance.
            layer.add_sample(Point2i(pixel_index % resolution.x, pixel_index / resolution.x),
                             counts[layer_index][pixel_index]);
        }

        const auto layer_path = image_path.parent_path() / (image_path.stem().string() + "-cascaded-frame-" +
                                                            std::to_string(layer_index) + ".png");
        layer.write_to_png(layer_path.string());
    }
    std::cout << firefly::layer_count << " cascaded framebuffers saved alongside `" << output_filename << "`\n";
}

__global__ static void megakernel_path_render(Film *film, uint8_t *gpu_frame_buffer, const int first_pixel,
                                              const int pixel_count, const int first_sample, const int sample_count,
                                              Sampler *samplers, const MegakernelPathIntegrator *integrator,
                                              const IntegratorBase *base, const Spectrum *cie_y, Real **counts,
                                              firefly::WeightRange **ranges, uint8_t *forced_outliers) {
    const auto resolution = film->get_resolution();
    const int local_pixel = threadIdx.x + blockIdx.x * blockDim.x;
    if (local_pixel >= pixel_count) {
        return;
    }

    const int pixel_index = first_pixel + local_pixel;
    const int x = pixel_index % resolution.x;
    const int y = pixel_index / resolution.x;
    const Point2i p_pixel(x, y);
    auto sampler = &samplers[pixel_index];

    for (int sample_idx = first_sample; sample_idx < first_sample + sample_count; ++sample_idx) {
        sampler->start_pixel_sample(pixel_index, sample_idx, 0);
        const auto camera_sample = sampler->get_camera_sample(p_pixel, base->filter);
        auto lambda = SampledWavelengths::sample_visible(sampler->get_1d());
        const auto ray = base->camera->generate_ray(camera_sample, sampler);
        const auto radiance_l = ray.weight * integrator->li(ray.ray, lambda, sampler);

        if (DEBUG_MODE && radiance_l.has_nan()) {
            printf("%s(): pixel(%d, %d), samples %d: has NAN\n", __func__, p_pixel.x, p_pixel.y, sample_idx);
        }

        film->add_sample(p_pixel, radiance_l, lambda, camera_sample.filter_weight);

        // Unfiltered CIE Y, including wavelength PDF compensation, in the same
        // units as RGBFilm's default CIE sensor (before averaging over camera samples).
        const Real luminance =
            (radiance_l * cie_y->sample(lambda)).safe_div(lambda.pdf_as_sampled_spectrum()).average();
        const auto sample = firefly::split(luminance);
        if (sample.lower_layer < 0) {
            continue;
        }

        // One thread owns each pixel. All positive finite camera samples,
        // including forced outliers, contribute one total histogram count.
        counts[sample.lower_layer][pixel_index] += sample.lower_count;
        counts[sample.lower_layer + 1][pixel_index] += sample.upper_count;
        ranges[sample.lower_layer][pixel_index].record(sample.lower_count);
        if (sample.forced_outlier) {
            forced_outliers[pixel_index] = 1;
        }
    }

    if (gpu_frame_buffer == nullptr) {
        return;
    }

    const auto rgb = film->get_pixel_rgb(p_pixel);
    const SRGBColorEncoding srgb_encoding;
    gpu_frame_buffer[pixel_index * 3 + 0] = rgb.has_nan() ? 0 : srgb_encoding.from_linear(rgb.r);
    gpu_frame_buffer[pixel_index * 3 + 1] = rgb.has_nan() ? 0 : srgb_encoding.from_linear(rgb.g);
    gpu_frame_buffer[pixel_index * 3 + 2] = rgb.has_nan() ? 0 : srgb_encoding.from_linear(rgb.b);
}

__global__ static void init_firefly_ranges(firefly::WeightRange **ranges, const int num_pixels) {
    const int pixel_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (pixel_index >= num_pixels) {
        return;
    }
    for (int layer = 0; layer < firefly::layer_count - 1; ++layer) {
        ranges[layer][pixel_index] = {};
    }
}

__global__ static void detect_fireflies(uint8_t *mask, Real **counts, firefly::WeightRange **ranges,
                                        const Real threshold, const Point2i resolution) {
    const int pixel_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (pixel_index >= resolution.x * resolution.y) {
        return;
    }

    mask[pixel_index] = firefly::is_candidate(counts, ranges, pixel_index, threshold, mask[pixel_index] != 0);
}

MegakernelPathIntegrator::MegakernelPathIntegrator(const int samples_per_pixel, const std::string &sampler_type,
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
    if (preview) {
        gl_helper.init("initializing", resolution);
    }

    // Fixed-size log-luminance count buffers covering alpha through the first
    // center above 1000. Per-interval observed weight ranges suffice to classify
    // every actual sample against the final frozen histogram without retracing.
    auto counts = local_allocator.allocate<Real *>(firefly::layer_count);
    auto ranges = local_allocator.allocate<firefly::WeightRange *>(firefly::layer_count - 1);
    for (int layer = 0; layer < firefly::layer_count; ++layer) {
        counts[layer] = local_allocator.allocate<Real>(num_pixels);
        CHECK_CUDA_ERROR(cudaMemset(counts[layer], 0, sizeof(Real) * num_pixels));
        if (layer + 1 < firefly::layer_count) {
            ranges[layer] = local_allocator.allocate<firefly::WeightRange>(num_pixels);
        }
    }
    auto mask = local_allocator.allocate<uint8_t>(num_pixels);
    CHECK_CUDA_ERROR(cudaMemset(mask, 0, sizeof(uint8_t) * num_pixels));
    constexpr int threads = 256;
    init_firefly_ranges<<<divide_and_ceil(num_pixels, threads), threads>>>(ranges, num_pixels);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Bound render batches independently of image size and total spp.
    constexpr int max_batch_samples = 16;
    constexpr int batch_capacity = 1 << 18;
    const int batch_samples = std::min(samples_per_pixel, max_batch_samples);
    const int batch_pixels = std::min(num_pixels, batch_capacity / batch_samples);

    size_t completed_samples = 0;
    const size_t total_samples = static_cast<size_t>(num_pixels) * samples_per_pixel;
    auto last_preview = std::chrono::steady_clock::now();
    for (int first_sample = 0; first_sample < samples_per_pixel; first_sample += batch_samples) {
        const int sample_count = std::min(batch_samples, samples_per_pixel - first_sample);
        for (int first_pixel = 0; first_pixel < num_pixels; first_pixel += batch_pixels) {
            const int pixel_count = std::min(batch_pixels, num_pixels - first_pixel);
            megakernel_path_render<<<divide_and_ceil(pixel_count, threads), threads>>>(
                film, gl_helper.gpu_frame_buffer, first_pixel, pixel_count, first_sample, sample_count, samplers, this,
                base, cie_y, counts, ranges, mask);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            completed_samples += static_cast<size_t>(pixel_count) * sample_count;
            const auto now = std::chrono::steady_clock::now();
            if (preview &&
                (now - last_preview >= std::chrono::milliseconds(200) || completed_samples == total_samples)) {
                gl_helper.draw_frame(GLHelper::assemble_title(double(completed_samples) / total_samples));
                last_preview = now;
            }
        }
    }

    // Classification sees a read-only snapshot containing all rendered spp.
    const Real threshold = firefly::tau(samples_per_pixel);
    detect_fireflies<<<divide_and_ceil(num_pixels, threads), threads>>>(mask, counts, ranges, threshold, resolution);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    size_t candidate_pixels = 0;
    for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
        const auto value = mask[pixel_index];
        firefly_mask.add_sample(Point2i(pixel_index % resolution.x, pixel_index / resolution.x), value);
        candidate_pixels += value;
    }
    printf("firefly detector (Bitterli 2019): %d layers (alpha %.3g, beta %d, tau %.6g, max %.6g), "
           "%zu/%d candidate pixels\n",
           firefly::layer_count, double(firefly::alpha), firefly::beta, double(threshold),
           double(firefly::max_luminance), candidate_pixels, num_pixels);

    // Export all count layers, including empty ones, before releasing the cascade.
    write_firefly_cascade(counts, resolution, output_filename);
}

PBRT_CPU_GPU
SampledSpectrum MegakernelPathIntegrator::evaluate_Li_volume(Real *depth, const Ray &primary_ray,
                                                             SampledWavelengths &lambda, Sampler *sampler,
                                                             const IntegratorBase *base, int max_depth,
                                                             bool regularize) {
    SampledSpectrum L = 0;
    SampledSpectrum beta = 1;
    bool specular_bounce = false;
    bool any_non_specular_bounces = false;

    pbrt::optional<Real> prev_direction_pdf;
    pbrt::optional<SurfaceInteraction> prev_interaction;
    SampledSpectrum multi_transmittance_pdf = 1;

    auto next_time_russian_roulette = start_russian_roulette;

    auto ray = primary_ray;

    Real path_depth = 0.0;
    // making depth float type to counter for contribution from material-less hit
    while (path_depth < max_depth && beta.is_positive()) {
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
            path_depth += IntegratorBase::interface_bounce_contribution;

            // in such case self-intersection is less worrying
            // because depth increases for every intersection

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

            break; // TODO wentao: this part is different from PBRT-v4, which does not break here
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

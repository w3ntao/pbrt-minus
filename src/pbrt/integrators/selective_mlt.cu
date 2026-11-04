#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/base/filter.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/sampler.h>
#include <pbrt/film/firefly_cascade.h>
#include <pbrt/film/grey_scale_film.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/integrators/megakernel_path.h>
#include <pbrt/integrators/mlt_bdpt.h>
#include <pbrt/integrators/selective_mlt.h>
#include <pbrt/samplers/stratified.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <stdexcept>
#include <thread>

namespace {

// The pilot only trains the detector; its radiance never enters the final Film.
__global__ void selective_pt_pilot(const Point2i resolution, const int spp, Sampler *samplers, const IntegratorBase *base,
                                   const Spectrum *cie_y, const int max_depth, const bool regularize,
                                   const firefly::CascadeView detector) {
    const int pixel_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (pixel_index >= resolution.x * resolution.y) {
        return;
    }

    const Point2i p_pixel(pixel_index % resolution.x, pixel_index / resolution.x);
    auto *sampler = &samplers[pixel_index];
    // Negative pilot IDs keep its jitter and stratum permutations independent
    // of production PT. They never index the image or detector buffers.
    for (int sample = 0; sample < spp; ++sample) {
        sampler->start_pixel_sample(~pixel_index, sample, 0);
        const auto camera_sample = sampler->get_camera_sample(p_pixel, base->filter);
        auto lambda = SampledWavelengths::sample_visible(sampler->get_1d());
        const auto ray = base->camera->generate_ray(camera_sample, sampler);
        const auto radiance = ray.weight * MegakernelPathIntegrator::evaluate_Li_volume(
                                               nullptr, ray.ray, lambda, sampler, base, max_depth, regularize, true);
        detector.record(pixel_index, radiance.y(lambda, cie_y));
    }
}

// One thread owns a listed production pixel for all of its samples.
__global__ void selective_pt_render(Film *film, const Point2i resolution, const int *pixel_indices,
                                    const int pixel_count, const int spp, const IntegratorBase *base,
                                    const int max_depth, const bool regularize) {
    const int thread_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_index >= pixel_count) {
        return;
    }
    const int pixel_index = pixel_indices[thread_index];

    const Point2i p_pixel(pixel_index % resolution.x, pixel_index / resolution.x);
    StratifiedSampler stratified(spp);
    Sampler sampler(&stratified);
    for (int sample = 0; sample < spp; ++sample) {
        sampler.start_pixel_sample(pixel_index, sample, 0);
        const auto camera_sample = sampler.get_camera_sample(p_pixel, base->filter);
        auto lambda = SampledWavelengths::sample_visible(sampler.get_1d());
        const auto ray = base->camera->generate_ray(camera_sample, &sampler);
        const auto radiance = ray.weight * MegakernelPathIntegrator::evaluate_Li_volume(
                                               nullptr, ray.ray, lambda, &sampler, base, max_depth, regularize, true);
        film->add_sample(pixel_index, radiance, lambda, camera_sample.filter_weight);
    }
}

void render_production_pt(Film *film, const Point2i resolution, const int spp, const IntegratorBase *base,
                          const int max_depth, const bool regularize, const uint8_t *frozen_mask) {
    const int num_pixels = resolution.x * resolution.y;
    GPUMemoryAllocator allocator;
    auto pixel_indices = allocator.allocate<int>(num_pixels);
    int pixel_count = 0;
    for (int pixel_index = 0; pixel_index < num_pixels; ++pixel_index) {
        // Frozen-mask pixels are handled by MLT.
        if (frozen_mask[pixel_index] == 0) {
            pixel_indices[pixel_count++] = pixel_index;
        }
    }
    if (pixel_count == 0) {
        return;
    }

    constexpr int threads = 256;
    selective_pt_render<<<divide_and_ceil(pixel_count, threads), threads>>>(
        film, resolution, pixel_indices, pixel_count, spp, base, max_depth, regularize);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace

SelectiveMLTIntegrator::SelectiveMLTIntegrator(const int samples_per_pixel, const ParameterDictionary &parameters,
                                               const IntegratorBase *base, const Film *film)
    : base(base), cie_y(parameters.global_spectra->cie_y), samples_per_pixel(samples_per_pixel),
      pilot_samples_per_pixel(parameters.get_integer("pilotspp", 64)),
      mutations_per_pixel(parameters.get_integer("mutationsperpixel", samples_per_pixel)), bootstrap_samples(0),
      num_chains(parameters.get_integer("numchains", std::min(16u, std::max(1u, std::thread::hardware_concurrency())))),
      max_depth(parameters.get_integer("maxdepth", 5)), regularize(parameters.get_bool("regularize", false)) {
    if (samples_per_pixel <= 0) {
        throw std::invalid_argument("selectivemlt requires a positive production PT spp");
    }
    if (StratifiedSampler::round_down_samples_per_pixel(samples_per_pixel) != samples_per_pixel) {
        throw std::invalid_argument("selectivemlt requires a perfect-square production PT spp for stratified sampling");
    }

    // At tiny production budgets the one-sample minimum takes precedence.
    const int pilot_limit = std::max(1, samples_per_pixel / 4);
    pilot_samples_per_pixel = std::clamp(pilot_samples_per_pixel, 1, pilot_limit);
    if (Sampler::default_sampler_type == Sampler::Type::stratified) {
        pilot_samples_per_pixel = StratifiedSampler::round_down_samples_per_pixel(pilot_samples_per_pixel);
    }

    const auto resolution = film->get_resolution();
    const auto num_pixels = static_cast<long long>(resolution.x) * resolution.y;
    if (resolution.x <= 0 || resolution.y <= 0 || num_pixels > std::numeric_limits<int>::max()) {
        throw std::runtime_error("selectivemlt requires a positive image size fitting a 32-bit pixel index");
    }
    bootstrap_samples = parameters.get_integer("bootstrapsamples", num_pixels);
    const Real sigma = parameters.get_float("sigma", 0.01);
    const Real large_step_probability = parameters.get_float("largestepprobability", 0.3);

    mlt = std::make_unique<MLTBDPTIntegrator>(base, max_depth, regularize, resolution, cie_y, mutations_per_pixel,
                                              num_chains, sigma, large_step_probability);
}

SelectiveMLTIntegrator::~SelectiveMLTIntegrator() = default;

double SelectiveMLTIntegrator::render(Film *film, GreyScaleFilm &film_firefly, GreyScaleFilm &sampling_density,
                                      const std::string &output_filename) const {
    const auto resolution = film->get_resolution();
    const int num_pixels = resolution.x * resolution.y;
    const auto image_path = std::filesystem::path(output_filename);
    const auto pt_path = image_path.parent_path() / (image_path.stem().string() + "-pt.png");
    const auto mlt_path = image_path.parent_path() / (image_path.stem().string() + "-mlt.png");
    GPUMemoryAllocator production_allocator;
    auto firefly_mask = production_allocator.allocate<uint8_t>(num_pixels);
    {
        GPUMemoryAllocator pilot_allocator;
        firefly::Cascade cascade(resolution, pilot_allocator);
        auto *pilot_samplers = Sampler::create_samplers(Sampler::default_sampler_type, pilot_samples_per_pixel,
                                                       num_pixels, pilot_allocator);
        printf("Selective MLT: %s PT pilot (%d spp; excluded from final image)\n",
               Sampler::sampler_type_to_string(Sampler::default_sampler_type).c_str(), pilot_samples_per_pixel);
        fflush(stdout);
        constexpr int threads = 256;
        selective_pt_pilot<<<divide_and_ceil(num_pixels, threads), threads>>>(
            resolution, pilot_samples_per_pixel, pilot_samplers, base, cie_y, max_depth, regularize, cascade.view());
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        cascade.finalize(film_firefly, pilot_samples_per_pixel, output_filename);
        const auto mask_path = image_path.parent_path() / (image_path.stem().string() + "-firefly.png");
        film_firefly.write_mask_to_png(mask_path.string());

        // Retain only the binary mask, freeing pilot histograms before either
        // production estimator runs. No detector updates follow this point.
        CHECK_CUDA_ERROR(
            cudaMemcpy(firefly_mask, cascade.view().mask, num_pixels * sizeof(uint8_t), cudaMemcpyDefault));
    }
    size_t selected = 0;
    for (int pixel = 0; pixel < num_pixels; ++pixel) {
        selected += firefly_mask[pixel] != 0;
    }

    if (selected <= 0) {
        REPORT_FATAL_ERROR();
    }

    printf("Selective MLT: frozen mask pixels: %.2f%%; production PT %d spp; MLT %d mutations/pixel\n",
           100.0 * selected / num_pixels, samples_per_pixel, mutations_per_pixel);
    fflush(stdout);

    double production_pt_seconds = 0;
    auto start = std::chrono::steady_clock::now();
    render_production_pt(film, resolution, samples_per_pixel, base, max_depth, regularize, firefly_mask);
    production_pt_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    film->write_to_png(pt_path.string(), 0);

    auto mlt_film = film->create_empty(production_allocator);
    double mlt_seconds = 0;
    double splat_scale = 0;

    start = std::chrono::steady_clock::now();
    const double normalization = mlt->render_firefly(mlt_film, sampling_density, firefly_mask, mutations_per_pixel,
                                                     bootstrap_samples, num_chains);
    mlt_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    splat_scale = normalization / mutations_per_pixel;

    auto total_time = production_pt_seconds + mlt_seconds;
    printf("Selective MLT time:\n");
    printf("     PT: %5.1f sec (%.1f%%)\n", production_pt_seconds, production_pt_seconds / total_time * 100);
    printf("    MLT: %5.1f sec (%.1f%%)\n", mlt_seconds, mlt_seconds / total_time * 100);
    fflush(stdout);
    mlt_film->write_to_png(mlt_path.string(), splat_scale);
    film->add_splats(*mlt_film);

    return splat_scale;
}

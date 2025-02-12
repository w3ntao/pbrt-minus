#include <numeric>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/sampler.h>
#include <pbrt/films/grey_scale_film.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/gui/gl_helper.h>
#include <pbrt/integrators/megakernel_path.h>
#include <pbrt/integrators/mlt_path.h>
#include <pbrt/samplers/mlt.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/global_spectra.h>

constexpr size_t NUM_MLT_PATH_SAMPLERS = 64 * 1024;
// large number of samplers: large number of shallow markov chains
// small number of samplers: small number of deep markov chains

struct MLTSample {
    PathSample path_sample;
    FloatType sampling_density;
    FloatType weight;

    PBRT_CPU_GPU
    MLTSample(const Point2f p_film, const SampledSpectrum &_radiance,
              const SampledWavelengths &_lambda, const FloatType _weight,
              const FloatType _sampling_density)
        : path_sample(PathSample(p_film, _radiance, _lambda)), weight(_weight),
          sampling_density(_sampling_density) {}
};

__global__ void build_bootstrap_samples(const uint num_paths_per_worker, double *luminance_per_path,
                                        const MLTPathIntegrator *mlt_integrator) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= NUM_MLT_PATH_SAMPLERS) {
        return;
    }

    auto sampler = &mlt_integrator->samplers[worker_idx];
    auto mlt_sampler = &mlt_integrator->mlt_samplers[worker_idx];

    for (int idx = 0; idx < num_paths_per_worker; idx++) {
        const auto path_idx = worker_idx * num_paths_per_worker + idx;

        mlt_sampler->init(path_idx);
        mlt_sampler->start_stream(0);

        const auto path_sample = mlt_integrator->generate_path_sample(sampler);

        const auto illuminance =
            mlt_integrator->compute_luminance(path_sample.radiance, path_sample.lambda);

        luminance_per_path[path_idx] = illuminance;
    }
}

__global__ void prepare_initial_state(PathSample *path_samples,
                                      const MLTPathIntegrator *mlt_integrator) {
    // this implementation is different from PBRT-v4:
    // in terms of selecting the 0th path samples (after bootstrap), PBRT-v4 did an importance
    // sampling from bootstrap (so multiple sampler might choose the same paths) but pbrt-minus
    // initiated each sampler with different bootstrap

    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= NUM_MLT_PATH_SAMPLERS) {
        return;
    }

    auto sampler = &mlt_integrator->samplers[worker_idx];
    auto mlt_sampler = &mlt_integrator->mlt_samplers[worker_idx];

    mlt_sampler->init(worker_idx);
    mlt_sampler->start_iteration();
    mlt_sampler->start_stream(0);

    path_samples[worker_idx] = mlt_integrator->generate_path_sample(sampler);
}

__global__ void wavefront_render(MLTSample *mlt_samples, const uint num_mutations,
                                 PathSample *path_samples, const MLTPathIntegrator *mlt_integrator,
                                 RNG *rngs) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num_mutations) {
        return;
    }

    auto sampler = &mlt_integrator->samplers[worker_idx];
    auto mlt_sampler = &mlt_integrator->mlt_samplers[worker_idx];
    auto rng = &rngs[worker_idx];

    mlt_sampler->start_iteration();
    mlt_sampler->start_stream(0);

    const auto proposed_path = mlt_integrator->generate_path_sample(sampler);
    const auto current_path = &path_samples[worker_idx];

    const auto proposed_c =
        mlt_integrator->compute_luminance(proposed_path.radiance, proposed_path.lambda);
    const auto current_c =
        mlt_integrator->compute_luminance(current_path->radiance, current_path->lambda);

    const auto sample_idx = worker_idx * 2;

    for (auto offset = 0; offset < 2; ++offset) {
        mlt_samples[sample_idx + offset].weight = 0;
        mlt_samples[sample_idx + offset].sampling_density = 0;
    }

    FloatType accept_prob = NAN;
    if (current_c == 0 && proposed_c == 0) {
        accept_prob = 0.5;

    } else if (current_c == 0) {
        accept_prob = 1;
        mlt_samples[sample_idx] = MLTSample(proposed_path.p_film, proposed_path.radiance,
                                            proposed_path.lambda, 1.0 / proposed_c, 1);

    } else if (proposed_c == 0) {
        accept_prob = 0;
        mlt_samples[sample_idx] = MLTSample(current_path->p_film, current_path->radiance,
                                            current_path->lambda, 1.0 / current_c, 1);
    } else {
        accept_prob = std::min<FloatType>(1.0, proposed_c / current_c);

        const FloatType proposed_path_weight = accept_prob / proposed_c;
        const FloatType current_path_weight = (1 - accept_prob) / current_c;

        mlt_samples[sample_idx + 0] =
            MLTSample(current_path->p_film, current_path->radiance, current_path->lambda,
                      current_path_weight, 1 - accept_prob);
        mlt_samples[sample_idx + 1] =
            MLTSample(proposed_path.p_film, proposed_path.radiance, proposed_path.lambda,
                      proposed_path_weight, accept_prob);
    }

    if (rng->uniform<FloatType>() < accept_prob) {
        *current_path = proposed_path;
        mlt_sampler->accept();

    } else {
        mlt_sampler->reject();
    }
}

MLTPathIntegrator *MLTPathIntegrator::create(const int mutations_per_pixel,
                                             const ParameterDictionary &parameters,
                                             const IntegratorBase *base,
                                             GPUMemoryAllocator &allocator) {
    auto integrator = allocator.allocate<MLTPathIntegrator>();

    integrator->base = base;
    integrator->mlt_samplers = allocator.allocate<MLTSampler>(NUM_MLT_PATH_SAMPLERS);
    integrator->samplers = allocator.allocate<Sampler>(NUM_MLT_PATH_SAMPLERS);

    const auto large_step_probability = parameters.get_float("largestepprobability", 0.3);
    const auto sigma = parameters.get_float("sigma", 0.01);

    for (uint idx = 0; idx < NUM_MLT_PATH_SAMPLERS; ++idx) {
        integrator->mlt_samplers[idx].setup_config(mutations_per_pixel, sigma,
                                                   large_step_probability, 1);
        integrator->samplers[idx].init(&integrator->mlt_samplers[idx]);
    }

    integrator->film_dimension = base->camera->get_camerabase()->resolution;
    integrator->cie_y = parameters.global_spectra->cie_y;

    integrator->max_depth = parameters.get_integer("maxdepth", 5);
    integrator->regularize = parameters.get_bool("regularize", false);

    return integrator;
}

PBRT_CPU_GPU
PathSample MLTPathIntegrator::generate_path_sample(Sampler *sampler) const {
    const auto lu = sampler->get_1d();
    auto lambda = SampledWavelengths::sample_visible(lu);

    const auto u = sampler->get_2d();
    const auto p_film = Point2f(u.x * film_dimension.x, u.y * film_dimension.y);
    const auto camera_sample = CameraSample(p_film, 1);

    const auto ray = base->camera->generate_ray(camera_sample, sampler);

    const auto radiance = ray.weight * MegakernelPathIntegrator::evaluate_li(
                                           ray.ray, lambda, base, sampler, max_depth, regularize);

    return PathSample(p_film, radiance, lambda);
}

double MLTPathIntegrator::render(Film *film, GreyScaleFilm &heat_map,
                                 const uint mutations_per_pixel, const bool preview) {
    GPUMemoryAllocator local_allocator;

    const auto num_paths_per_worker =
        divide_and_ceil<int>(film_dimension.x * film_dimension.y, NUM_MLT_PATH_SAMPLERS);

    const auto num_bootstrap_paths = num_paths_per_worker * NUM_MLT_PATH_SAMPLERS;

    if (num_paths_per_worker <= 0) {
        REPORT_FATAL_ERROR();
    }

    auto path_samples = local_allocator.allocate<PathSample>(NUM_MLT_PATH_SAMPLERS);
    auto luminance_per_path = local_allocator.allocate<double>(num_bootstrap_paths);

    constexpr uint threads = 64;
    const uint blocks = divide_and_ceil<uint>(NUM_MLT_PATH_SAMPLERS, threads);
    build_bootstrap_samples<<<blocks, threads>>>(num_paths_per_worker, luminance_per_path, this);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    const double sum_luminance =
        std::accumulate(luminance_per_path + 0, luminance_per_path + num_bootstrap_paths, 0.0);

    const double brightness = sum_luminance / num_bootstrap_paths;

    const long long total_mutations =
        static_cast<long long>(mutations_per_pixel) * film_dimension.x * film_dimension.y;

    const auto total_pass = divide_and_ceil<long long>(total_mutations, NUM_MLT_PATH_SAMPLERS);

    printf("MLT-PATH:\n");
    printf("    number of bootstrap paths: %lu\n", num_bootstrap_paths);
    printf("    MLT samplers: %lu\n", NUM_MLT_PATH_SAMPLERS);
    printf("    mutations per samplers: %.1f\n", double(total_mutations) / NUM_MLT_PATH_SAMPLERS);
    printf("    brightness: %.6f\n", brightness);

    if (brightness <= 0.0) {
        REPORT_FATAL_ERROR();
    }

    auto mlt_samples = local_allocator.allocate<MLTSample>(2 * NUM_MLT_PATH_SAMPLERS);

    GLHelper gl_helper;
    if (preview) {
        gl_helper.init("initializing", film_dimension);
    }

    auto rngs = local_allocator.allocate<RNG>(NUM_MLT_PATH_SAMPLERS);

    for (uint idx = 0; idx < NUM_MLT_PATH_SAMPLERS; ++idx) {
        rngs[idx].set_sequence(idx + NUM_MLT_PATH_SAMPLERS);
    }

    prepare_initial_state<<<blocks, threads>>>(path_samples, this);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    long long accumulate_samples = 0; // this is for debugging and verification
    for (uint pass = 0; pass < total_pass; ++pass) {
        const uint num_mutations = pass == total_pass - 1
                                       ? total_mutations - (total_pass - 1) * NUM_MLT_PATH_SAMPLERS
                                       : NUM_MLT_PATH_SAMPLERS;

        wavefront_render<<<blocks, threads>>>(mlt_samples, num_mutations, path_samples, this, rngs);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        for (uint idx = 0; idx < num_mutations * 2; ++idx) {
            accumulate_samples += 1;
            const auto path_sample = &mlt_samples[idx].path_sample;
            const auto p_film = path_sample->p_film;
            const auto weight = mlt_samples[idx].weight;
            const auto sampling_density = mlt_samples[idx].sampling_density;

            if (weight > 0) {
                film->add_splat(p_film, path_sample->radiance * weight, path_sample->lambda);
            }

            auto p_discrete = (p_film + Vector2f(0.5, 0.5)).floor();
            p_discrete.x = clamp<int>(p_discrete.x, 0, film_dimension.x - 1);
            p_discrete.y = clamp<int>(p_discrete.y, 0, film_dimension.y - 1);

            heat_map.add_sample(p_discrete, sampling_density);
        }

        if (preview) {
            film->copy_to_frame_buffer(gl_helper.gpu_frame_buffer,
                                       brightness / mutations_per_pixel);
            gl_helper.draw_frame(GLHelper::assemble_title(FloatType(pass + 1) / total_pass));
        }
    }

    if (accumulate_samples != total_mutations * 2) {
        REPORT_FATAL_ERROR();
    }

    return brightness;
}

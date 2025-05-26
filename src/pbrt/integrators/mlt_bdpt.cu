#include <numeric>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/sampler.h>
#include <pbrt/cameras/perspective.h>
#include <pbrt/film/grey_scale_film.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/gui/gl_helper.h>
#include <pbrt/integrators/bdpt.h>
#include <pbrt/integrators/mlt_bdpt.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/global_spectra.h>

constexpr size_t NUM_MLT_SAMPLERS = 8 * 1024;
constexpr size_t NUM_BOOT_STRAP = 1000000;

struct MLTSample {
    BDPTPathSample path_sample;
    Real weight;
    Real sampling_density;

    PBRT_CPU_GPU
    MLTSample(const Point2f &p_film, const SampledSpectrum &_radiance,
              const SampledWavelengths &_lambda, const Real _weight, const Real _sampling_density)
        : path_sample(BDPTPathSample(p_film, _radiance, _lambda)), weight(_weight),
          sampling_density(_sampling_density) {}
};

__global__ void build_bootstrap_samples(double *luminance_per_path, const int repeat,
                                        Vertex *global_camera_vertices,
                                        Vertex *global_light_vertices,
                                        const MLTBDPTIntegrator *mlt_bdpt_integrator) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= NUM_MLT_SAMPLERS) {
        return;
    }

    auto sampler = &mlt_bdpt_integrator->samplers[worker_idx];
    auto mlt_sampler = &mlt_bdpt_integrator->mlt_samplers[worker_idx];

    const auto max_depth = mlt_bdpt_integrator->config->max_depth;

    auto local_camera_vertices = &global_camera_vertices[worker_idx * (max_depth + 2)];
    auto local_light_vertices = &global_light_vertices[worker_idx * (max_depth + 1)];

    for (int idx = 0; idx < repeat; ++idx) {
        for (int depth = 0; depth < max_depth + 1; ++depth) {
            mlt_sampler->init(worker_idx * (max_depth + 1) * repeat + idx * repeat + depth);
            mlt_sampler->advance(NUM_MLT_SAMPLERS);

            auto path_sample = mlt_bdpt_integrator->li(depth, sampler, local_camera_vertices,
                                                       local_light_vertices);
            const auto illuminance =
                mlt_bdpt_integrator->compute_luminance(path_sample.radiance, path_sample.lambda);

            luminance_per_path[worker_idx * (max_depth + 1) + depth] += illuminance;
        }
    }
}

MLTBDPTIntegrator *MLTBDPTIntegrator::create(const int mutations_per_pixel,
                                             const ParameterDictionary &parameters,
                                             const IntegratorBase *base,
                                             GPUMemoryAllocator &allocator) {
    auto integrator = allocator.allocate<MLTBDPTIntegrator>();

    auto config = allocator.allocate<BDPTConfig>();

    config->base = base;
    config->max_depth = parameters.get_integer("maxdepth", 5);
    config->regularize = parameters.get_bool("regularize", false);
    config->film_sample_size = config->max_depth * NUM_MLT_SAMPLERS;

    integrator->config = config;

    integrator->mlt_samplers = allocator.allocate<MLTSampler>(NUM_MLT_SAMPLERS);
    integrator->samplers = allocator.allocate<Sampler>(NUM_MLT_SAMPLERS);

    const auto large_step_probability = parameters.get_float("largestepprobability", 0.3);
    const auto sigma = parameters.get_float("sigma", 0.01);

    for (uint idx = 0; idx < NUM_MLT_SAMPLERS; ++idx) {
        integrator->mlt_samplers[idx].setup_config(mutations_per_pixel, sigma,
                                                   large_step_probability, 3);
        REPORT_FATAL_ERROR();
        // TODO: implement me
        // integrator->samplers[idx].init(&integrator->mlt_samplers[idx]);
    }

    integrator->film_dimension = base->camera->get_camerabase()->resolution;
    integrator->cie_y = parameters.global_spectra->cie_y;

    return integrator;
}

PBRT_GPU
BDPTPathSample MLTBDPTIntegrator::li(const int depth, Sampler *sampler, Vertex *camera_vertices,
                                     Vertex *light_vertices) const {
    if (config->base->light_num == 0) {
        return BDPTPathSample::zero();
    }

    auto mlt_sampler = sampler->get_mlt_sampler();

    mlt_sampler->start_stream(cameraStreamIndex);

    // Determine the number of available strategies and pick a specific one
    int s, t, nStrategies;
    if (depth == 0) {
        nStrategies = 1;
        s = 0;
        t = 2;
    } else {
        nStrategies = depth + 2;
        s = std::min<int>(sampler->get_1d() * nStrategies, nStrategies - 1);
        t = nStrategies - s;
    }

    const auto lu = sampler->get_1d();
    auto lambda = SampledWavelengths::sample_visible(lu);

    const auto u = sampler->get_2d();
    auto p_film = Point2f(u.x * film_dimension.x, u.y * film_dimension.y);
    const auto camera_sample = CameraSample(p_film, 1);

    const auto camera_ray = config->base->camera->generate_ray(camera_sample, sampler);

    if (BDPTIntegrator::generate_camera_subpath(camera_ray.ray, lambda, sampler, t, camera_vertices,
                                                config) != t) {
        return BDPTPathSample::zero();
    }

    mlt_sampler->start_stream(lightStreamIndex);
    if (BDPTIntegrator::generate_light_subpath(lambda, sampler, s, light_vertices, config) != s) {
        return BDPTPathSample::zero();
    }

    // Execute connection strategy and return the radiance estimate
    mlt_sampler->start_stream(connectionStreamIndex);
    pbrt::optional<Point2f> p_raster_new;
    const auto radiance_l = BDPTIntegrator::connect_bdpt(lambda, light_vertices, camera_vertices, s,
                                                         t, sampler, &p_raster_new, config) *
                            nStrategies;

    if (p_raster_new.has_value()) {
        p_film = p_raster_new.value();
    }

    return BDPTPathSample(p_film, radiance_l, lambda);
}

__global__ void prepare_initial_state(BDPTPathSample *path_samples, Vertex *global_camera_vertices,
                                      Vertex *global_light_vertices,
                                      const MLTBDPTIntegrator *mlt_bdpt_integrator) {
    // this implementation is different from PBRT-v4:
    // in terms of selecting the 0th path samples (after bootstrap), PBRT-v4 did an importance
    // sampling from bootstrap (so multiple sampler might choose the same paths) but pbrt-minus
    // initiated each sampler with different bootstrap

    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= NUM_MLT_SAMPLERS) {
        return;
    }

    const auto max_depth = mlt_bdpt_integrator->config->max_depth;

    auto local_camera_vertices = &global_camera_vertices[worker_idx * (max_depth + 2)];
    auto local_light_vertices = &global_light_vertices[worker_idx * (max_depth + 1)];

    auto sampler = &mlt_bdpt_integrator->samplers[worker_idx];
    auto mlt_sampler = &mlt_bdpt_integrator->mlt_samplers[worker_idx];

    mlt_sampler->init(worker_idx);
    mlt_sampler->start_iteration();

    mlt_sampler->start_stream(MLTBDPTIntegrator::connectionStreamIndex);
    const auto depth = clamp<int>(mlt_sampler->get_1d() * (max_depth + 1), 0, max_depth);
    // unlike PBRT-v4: I get depth from MLT sampler rather than a previously computed AliasTable

    path_samples[worker_idx] =
        mlt_bdpt_integrator->li(depth, sampler, local_camera_vertices, local_light_vertices);
}

__global__ void wavefront_render(MLTSample *mlt_samples, BDPTPathSample *path_samples,
                                 const uint num_mutations, Vertex *global_camera_vertices,
                                 Vertex *global_light_vertices, RNG *rngs,
                                 const MLTBDPTIntegrator *mlt_bdpt_integrator) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num_mutations) {
        return;
    }

    auto sampler = &mlt_bdpt_integrator->samplers[worker_idx];
    auto mlt_sampler = &mlt_bdpt_integrator->mlt_samplers[worker_idx];
    auto rng = &rngs[worker_idx];

    mlt_sampler->start_iteration();

    const auto max_depth = mlt_bdpt_integrator->config->max_depth;

    auto local_camera_vertices = &global_camera_vertices[worker_idx * (max_depth + 2)];
    auto local_light_vertices = &global_light_vertices[worker_idx * (max_depth + 1)];

    mlt_sampler->start_stream(MLTBDPTIntegrator::connectionStreamIndex);
    const auto depth = clamp<int>(mlt_sampler->get_1d() * (max_depth + 1), 0, max_depth);
    // unlike PBRT-v4: I get depth from MLT sampler rather than a previously computed AliasTable

    auto proposed_path =
        mlt_bdpt_integrator->li(depth, sampler, local_camera_vertices, local_light_vertices);
    auto current_path = &path_samples[worker_idx];

    auto proposed_c =
        mlt_bdpt_integrator->compute_luminance(proposed_path.radiance, proposed_path.lambda);

    const auto current_c =
        mlt_bdpt_integrator->compute_luminance(current_path->radiance, current_path->lambda);

    const auto sample_idx = worker_idx * 2;

    for (auto offset = 0; offset < 2; ++offset) {
        mlt_samples[sample_idx + offset].weight = 0;
        mlt_samples[sample_idx + offset].sampling_density = 0;
    }

    Real accept_prob = NAN;
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
        accept_prob = std::min<Real>(1.0, proposed_c / current_c);

        const Real proposed_path_weight = accept_prob / proposed_c;
        const Real current_path_weight = (1 - accept_prob) / current_c;

        mlt_samples[sample_idx + 0] =
            MLTSample(current_path->p_film, current_path->radiance, current_path->lambda,
                      current_path_weight, 1 - accept_prob);
        mlt_samples[sample_idx + 1] =
            MLTSample(proposed_path.p_film, proposed_path.radiance, proposed_path.lambda,
                      proposed_path_weight, accept_prob);
    }

    if (rng->uniform<Real>() < accept_prob) {
        *current_path = proposed_path;
        mlt_sampler->accept();
    } else {
        mlt_sampler->reject();
    }
}

double MLTBDPTIntegrator::render(Film *film, GreyScaleFilm &heat_map, uint mutations_per_pixel,
                                 bool preview) {
    GPUMemoryAllocator local_allocator;

    const auto num_bootstrap_paths = NUM_MLT_SAMPLERS * (config->max_depth + 1);

    auto luminance_per_path = local_allocator.allocate<double>(num_bootstrap_paths);
    for (uint idx = 0; idx < num_bootstrap_paths; ++idx) {
        luminance_per_path[idx] = 0;
    }

    auto global_camera_vertices =
        local_allocator.allocate<Vertex>(NUM_MLT_SAMPLERS * (config->max_depth + 2));
    auto global_light_vertices =
        local_allocator.allocate<Vertex>(NUM_MLT_SAMPLERS * (config->max_depth + 1));

    const auto repeat =
        divide_and_ceil<int>(NUM_BOOT_STRAP, NUM_MLT_SAMPLERS * (config->max_depth + 1));
    // to make sure at least 1 million bootstrap samples recorded

    constexpr uint threads = 64;
    const uint blocks = divide_and_ceil<uint>(NUM_MLT_SAMPLERS, threads);
    build_bootstrap_samples<<<blocks, threads>>>(luminance_per_path, repeat, global_camera_vertices,
                                                 global_light_vertices, this);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    const auto brightness =
        Real(config->max_depth + 1) / (num_bootstrap_paths * repeat) *
        std::accumulate(luminance_per_path, luminance_per_path + num_bootstrap_paths, 0.);

    if (brightness <= 0 || std::isnan(brightness) || std::isinf(brightness)) {
        REPORT_FATAL_ERROR();
    }

    const long long total_mutations =
        static_cast<long long>(mutations_per_pixel) * film_dimension.x * film_dimension.y;

    const auto total_pass = divide_and_ceil<long long>(total_mutations, NUM_MLT_SAMPLERS);

    printf("MLT-BDPT:\n");
    printf("    number of bootstrap paths: %lu\n", num_bootstrap_paths);
    printf("    MLT samplers: %lu\n", NUM_MLT_SAMPLERS);
    printf("    mutations per samplers: %.1f\n", double(total_mutations) / NUM_MLT_SAMPLERS);
    printf("    brightness: %.6f\n", brightness);

    auto rngs = local_allocator.allocate<RNG>(NUM_MLT_SAMPLERS);
    for (uint idx = 0; idx < NUM_MLT_SAMPLERS; ++idx) {
        rngs[idx].set_sequence(idx + NUM_MLT_SAMPLERS);
    }

    auto path_samples = local_allocator.allocate<BDPTPathSample>(NUM_MLT_SAMPLERS);

    prepare_initial_state<<<blocks, threads>>>(path_samples, global_camera_vertices,
                                               global_light_vertices, this);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    auto mlt_samples = local_allocator.allocate<MLTSample>(NUM_MLT_SAMPLERS * 2);

    GLHelper gl_helper;
    if (preview) {
        gl_helper.init("initializing", film_dimension);
    }

    long long accumulate_samples = 0; // this is for debugging and verification
    for (uint pass = 0; pass < total_pass; ++pass) {
        const uint num_mutations = pass == total_pass - 1
                                       ? total_mutations - (total_pass - 1) * NUM_MLT_SAMPLERS
                                       : NUM_MLT_SAMPLERS;

        wavefront_render<<<blocks, threads>>>(mlt_samples, path_samples, num_mutations,
                                              global_camera_vertices, global_light_vertices, rngs,
                                              this);
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

            if (sampling_density > 0) {
                auto p_discrete = (p_film + Vector2f(0.5, 0.5)).floor();
                p_discrete.x = clamp<int>(p_discrete.x, 0, film_dimension.x - 1);
                p_discrete.y = clamp<int>(p_discrete.y, 0, film_dimension.y - 1);

                heat_map.add_sample(p_discrete, sampling_density);
            }
        }

        if (preview) {
            film->copy_to_frame_buffer(gl_helper.gpu_frame_buffer,
                                       brightness / mutations_per_pixel);

            gl_helper.draw_frame(GLHelper::assemble_title(Real(pass + 1) / total_pass));
        }
    }

    if (accumulate_samples != total_mutations * 2) {
        REPORT_FATAL_ERROR();
    }

    return brightness;
}

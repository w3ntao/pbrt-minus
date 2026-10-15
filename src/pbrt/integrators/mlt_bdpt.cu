#include <algorithm>
#include <cmath>
#include <numeric>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/sampler.h>
#include <pbrt/cameras/perspective.h>
#include <pbrt/film/grey_scale_film.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/integrators/bdpt.h>
#include <pbrt/integrators/mlt_bdpt.h>
#include <pbrt/samplers/mlt.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/util/thread_pool.h>
#include <thread>
#include <vector>

BDPTPathSample MLTBDPTIntegrator::generate_path_sample(MLTSampler &mlt_sampler) const {
    // Reserve the first connection-stream dimension for depth selection.
    mlt_sampler.start_stream(connectionStreamIndex);
    const auto depth = clamp<int>(mlt_sampler.get_1d() * (config->max_depth + 1), 0, config->max_depth);
    Sampler sampler(&mlt_sampler);
    return li(depth, &sampler);
}

BDPTPathSample MLTBDPTIntegrator::generate_bootstrap_path(const long bootstrap_index, MLTSampler &mlt_sampler) const {
    mlt_sampler.setup_config(mutations_per_pixel, sigma, large_step_probability, nSampleStreams);
    mlt_sampler.init(bootstrap_index);
    // Bootstrap generation and replay must consume the same RNG sequence.
    mlt_sampler.start_iteration();
    mlt_sampler.large_step = true;
    const auto path = generate_path_sample(mlt_sampler);
    // Commit iteration 1 so unused dimensions get a uniform value on first use.
    mlt_sampler.accept();
    return path;
}

void MLTBDPTIntegrator::build_bootstrap_samples(const int worker_idx, const int num_paths_per_worker,
                                                std::vector<double> &luminance_per_path) const {
    for (int idx = 0; idx < num_paths_per_worker; ++idx) {
        const auto path_idx = static_cast<long>(worker_idx) * num_paths_per_worker + idx;
        // init() does not clear PrimarySample state; each bootstrap needs a fresh sampler.
        MLTSampler mlt_sampler;
        const auto path = generate_bootstrap_path(path_idx, mlt_sampler);
        const auto luminance = compute_luminance(path.radiance, path.lambda);
        if (luminance < 0 || !std::isfinite(luminance)) {
            printf("MLT-BDPT: invalid bootstrap luminance at path %ld\n", path_idx);
            REPORT_FATAL_ERROR();
        }
        luminance_per_path[path_idx] = luminance;
    }
}

MLTBDPTIntegrator *MLTBDPTIntegrator::create(const int mutations_per_pixel, const ParameterDictionary &parameters,
                                             const IntegratorBase *base, GPUMemoryAllocator &allocator) {
    const auto max_depth = parameters.get_integer("maxdepth", 5);
    const auto regularize = parameters.get_bool("regularize", false);
    // hardware_concurrency() may return zero when the thread count is unknown.
    const int num_samplers = std::thread::hardware_concurrency();
    const auto film_sample_size = max_depth * num_samplers;

    const auto config = allocator.create<BDPTConfig>(base, max_depth, film_sample_size, regularize);

    const auto large_step_probability = parameters.get_float("largestepprobability", 0.3);
    const auto sigma = parameters.get_float("sigma", 0.01);

    return allocator.create<MLTBDPTIntegrator>(config, base->camera->get_camera_base()->resolution,
                                               parameters.global_spectra->cie_y, mutations_per_pixel, num_samplers,
                                               sigma, large_step_probability);
}

BDPTPathSample MLTBDPTIntegrator::li(const int depth, Sampler *sampler) const {
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

    std::vector<Vertex> camera_vertices(t);
    std::vector<Vertex> light_vertices(s);

    if (BDPTIntegrator::generate_camera_subpath(camera_ray.ray, lambda, sampler, t, camera_vertices.data(), config) !=
        t) {
        return BDPTPathSample::zero();
    }

    mlt_sampler->start_stream(lightStreamIndex);
    if (BDPTIntegrator::generate_light_subpath(lambda, sampler, s, light_vertices.data(), config) != s) {
        return BDPTPathSample::zero();
    }

    // Execute connection strategy and return the radiance estimate
    mlt_sampler->start_stream(connectionStreamIndex);
    // Dimension 0 already selected depth; reading it again would overwrite its backup.
    mlt_sampler->sample_index = 1;
    pbrt::optional<Point2f> p_raster_new;
    const auto radiance_l =
        BDPTIntegrator::connect_bdpt(lambda, light_vertices, camera_vertices, s, t, sampler, &p_raster_new, config) *
        nStrategies;

    if (p_raster_new.has_value()) {
        p_film = p_raster_new.value();
    }

    return BDPTPathSample(p_film, radiance_l, lambda);
}

void MLTBDPTIntegrator::mutate_chain(const int worker_idx, Film *sub_film, GreyScaleFilm &sub_heat_map,
                                     const long long num_mutations, const std::vector<double> &luminance_per_path,
                                     const std::vector<double> &bootstrap_cdf) const {
    // Independent chain RNGs can select the same bootstrap path and still diverge.
    RNG rng(worker_idx + num_samplers);
    const double sum_luminance = bootstrap_cdf.back();
    const double u = std::min(rng.uniform<double>() * sum_luminance, std::nextafter(sum_luminance, 0.0));
    const auto selected = std::upper_bound(bootstrap_cdf.begin(), bootstrap_cdf.end(), u);
    if (selected == bootstrap_cdf.end()) {
        REPORT_FATAL_ERROR();
    }

    const auto bootstrap_index = static_cast<long>(selected - bootstrap_cdf.begin());
    MLTSampler mlt_sampler;
    auto current_path = generate_bootstrap_path(bootstrap_index, mlt_sampler);
    const auto bootstrap_luminance = compute_luminance(current_path.radiance, current_path.lambda);
    if (bootstrap_luminance <= 0 || bootstrap_luminance != luminance_per_path[bootstrap_index]) {
        printf("MLT-BDPT: failed to replay bootstrap path %ld\n", bootstrap_index);
        REPORT_FATAL_ERROR();
    }

    const auto sub_film_dimension = sub_film->get_resolution();

    const auto add_sample = [&](const BDPTPathSample &path, const Real weight, const Real sampling_density) {
        if (weight > 0) {
            sub_film->add_splat(path.p_film, path.radiance * weight, path.lambda);
        }

        if (sampling_density > 0) {
            auto p_discrete = (path.p_film + Vector2f(0.5, 0.5)).floor();
            p_discrete.x = clamp<int>(p_discrete.x, 0, sub_film_dimension.x - 1);
            p_discrete.y = clamp<int>(p_discrete.y, 0, sub_film_dimension.y - 1);

            sub_heat_map.add_sample(p_discrete, sampling_density);
        }
    };

    for (long long mutation_idx = 0; mutation_idx < num_mutations; ++mutation_idx) {
        mlt_sampler.start_iteration();
        auto proposed_path = generate_path_sample(mlt_sampler);

        auto proposed_c = compute_luminance(proposed_path.radiance, proposed_path.lambda);

        const auto current_c = compute_luminance(current_path.radiance, current_path.lambda);

        Real accept_prob = NAN;
        if (current_c == 0 && proposed_c == 0) {
            accept_prob = 0.5;
        } else if (current_c == 0) {
            accept_prob = 1;
            add_sample(proposed_path, 1.0 / proposed_c, 1);
        } else if (proposed_c == 0) {
            accept_prob = 0;
            add_sample(current_path, 1.0 / current_c, 1);
        } else {
            accept_prob = std::min<Real>(1.0, proposed_c / current_c);

            const Real proposed_path_weight = accept_prob / proposed_c;
            const Real current_path_weight = (1 - accept_prob) / current_c;

            add_sample(current_path, current_path_weight, 1 - accept_prob);
            add_sample(proposed_path, proposed_path_weight, accept_prob);
        }

        if (rng.uniform<Real>() < accept_prob) {
            current_path = proposed_path;
            mlt_sampler.accept();
        } else {
            mlt_sampler.reject();
        }
    }
}

double MLTBDPTIntegrator::render(Film *film, GreyScaleFilm &heat_map, int mutations_per_pixel) {
    ThreadPool thread_pool(num_samplers);
    const auto num_paths_per_worker = divide_and_ceil<int>(film_dimension.x * film_dimension.y, num_samplers);
    const auto num_bootstrap_paths = static_cast<size_t>(num_paths_per_worker) * num_samplers;

    std::vector<double> luminance_per_path(num_bootstrap_paths);
    thread_pool.parallel_execute(0, num_samplers, [&](const int worker_idx) {
        build_bootstrap_samples(worker_idx, num_paths_per_worker, luminance_per_path);
    });

    const auto num_nonzero_paths = static_cast<size_t>(std::count_if(
        luminance_per_path.begin(), luminance_per_path.end(), [](const double luminance) { return luminance > 0.0; }));
    std::vector<double> bootstrap_cdf(num_bootstrap_paths);
    std::partial_sum(luminance_per_path.begin(), luminance_per_path.end(), bootstrap_cdf.begin());
    const double sum_luminance = bootstrap_cdf.back();
    // li() compensates for strategy sampling; uniform depth selection needs this factor too.
    const double luminance = (config->max_depth + 1) * (sum_luminance / num_bootstrap_paths);

    const long long total_mutations = static_cast<long long>(mutations_per_pixel) * film_dimension.x * film_dimension.y;

    std::vector<long long> num_mutations(num_samplers, total_mutations / num_samplers);
    for (int idx = 0; idx < total_mutations % num_samplers; ++idx) {
        ++num_mutations[idx];
    }

    printf("MLT-BDPT:\n");
    printf("    bootstrap paths: %zu (nonzero: %.1f%%)\n", num_bootstrap_paths,
           100.0 * num_nonzero_paths / num_bootstrap_paths);
    printf("    MLT samplers: %d\n", num_samplers);
    printf("    mutations per samplers: %.1f\n", double(total_mutations) / num_samplers);
    printf("    luminance: %.6f\n", luminance);

    if (luminance <= 0 || !std::isfinite(luminance) || num_nonzero_paths == 0) {
        printf("MLT-BDPT: bootstrap requires a positive, finite total luminance\n");
        REPORT_FATAL_ERROR();
    }

    GPUMemoryAllocator allocator;
    std::vector<Film *> sub_films(num_samplers);
    std::vector<GreyScaleFilm> sub_heat_maps;
    for (int idx = 0; idx < num_samplers; ++idx) {
        sub_films[idx] = film->create_empty(allocator);
        sub_heat_maps.emplace_back(film_dimension);
    }

    if (std::accumulate(num_mutations.begin(), num_mutations.end(), 0LL) != total_mutations) {
        REPORT_FATAL_ERROR();
    }

    thread_pool.parallel_execute(0, num_samplers, [&](const int worker_idx) {
        mutate_chain(worker_idx, sub_films[worker_idx], sub_heat_maps[worker_idx], num_mutations[worker_idx],
                     luminance_per_path, bootstrap_cdf);
    });

    // Merge raw chain contributions once; output normalization still uses luminance / mutations_per_pixel.
    for (int idx = 0; idx < num_samplers; ++idx) {
        film->add_splats(*sub_films[idx]);
        heat_map.add_samples(sub_heat_maps[idx]);
    }

    return luminance;
}

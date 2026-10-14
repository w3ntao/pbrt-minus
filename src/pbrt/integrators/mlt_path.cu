#include <algorithm>
#include <memory>
#include <numeric>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/sampler.h>
#include <pbrt/cameras/perspective.h>
#include <pbrt/film/grey_scale_film.h>
#include <pbrt/integrators/megakernel_path.h>
#include <pbrt/integrators/mlt_path.h>
#include <pbrt/samplers/mlt.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/util/thread_pool.h>
#include <thread>
#include <vector>

struct PathSample {
    Point2f p_film;
    SampledSpectrum radiance;
    SampledWavelengths lambda;
    Real depth;

    PathSample(const Point2f &_p_film, const SampledSpectrum &_radiance, const SampledWavelengths &_lambda, Real _depth)
        : p_film(_p_film), radiance(_radiance), lambda(_lambda), depth(_depth) {}
};

Real MLTPathIntegrator::compute_importance(const PathSample &path) const {
    // the significance of 0.001:
    // if the depth is 0 and an area light hit: we change the depth to 0.001 to prevent degenerating
    // yet also limit the contribution

    return (path.depth + 0.001) / max_depth * compute_luminance(path.radiance, path.lambda);
}

static void build_bootstrap_samples(const int worker_idx, const int num_paths_per_worker,
                                    std::vector<double> &importance_per_path, const MLTPathIntegrator *mlt_integrator) {
    MLTSampler mlt_sampler;
    mlt_sampler.setup_config(mlt_integrator->mutations_per_pixel, mlt_integrator->sigma,
                             mlt_integrator->large_step_probability, 1);
    Sampler sampler(&mlt_sampler);

    for (int idx = 0; idx < num_paths_per_worker; idx++) {
        const auto path_idx = worker_idx * num_paths_per_worker + idx;

        mlt_sampler.init(path_idx);
        mlt_sampler.start_stream(0);

        const auto path_sample = mlt_integrator->generate_path_sample(&sampler);

        importance_per_path[path_idx] = mlt_integrator->compute_importance(path_sample);
    }
}

void mutate_chain(const int seed, Film *chain_film, GreyScaleFilm &chain_heat_map,
                  const MLTPathIntegrator *mlt_integrator) {
    MLTSampler mlt_sampler;
    mlt_sampler.setup_config(mlt_integrator->mutations_per_pixel, mlt_integrator->sigma,
                             mlt_integrator->large_step_probability, 1);
    Sampler sampler(&mlt_sampler);
    RNG rng(seed);

    // Initialize each chain with a uniform large step before applying mutations.
    // The initial paths use distinct seeds rather than bootstrap importance sampling.
    mlt_sampler.init(seed);
    mlt_sampler.start_iteration();
    mlt_sampler.large_step = true;
    mlt_sampler.start_stream(0);
    auto current_path = mlt_integrator->generate_path_sample(&sampler);
    // Accept iteration 1 so unused dimensions get a uniform value on first use.
    auto current_c = mlt_integrator->compute_importance(current_path);

    mlt_sampler.accept();

    const auto add_sample = [&](const PathSample &path, const Real weight, const Real sampling_density) {
        if (weight > 0) {
            chain_film->add_splat(path.p_film, path.radiance * weight, path.lambda);
        }
        if (sampling_density > 0) {
            auto p_discrete = (path.p_film + Vector2f(0.5, 0.5)).floor();
            p_discrete.x = clamp<int>(p_discrete.x, 0, mlt_integrator->film_dimension.x - 1);
            p_discrete.y = clamp<int>(p_discrete.y, 0, mlt_integrator->film_dimension.y - 1);
            chain_heat_map.add_sample(p_discrete, sampling_density);
        }
    };

    for (long long idx = 0; idx < mlt_integrator->mutations_per_sampler; ++idx) {
        mlt_sampler.start_iteration();
        mlt_sampler.start_stream(0);

        const auto proposed_path = mlt_integrator->generate_path_sample(&sampler);
        const auto proposed_c = mlt_integrator->compute_importance(proposed_path);

        Real accept_prob;
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
            current_c = proposed_c;

            mlt_sampler.accept();

        } else {
            mlt_sampler.reject();
        }
    }
}

std::unique_ptr<MLTPathIntegrator> MLTPathIntegrator::create(const int mutations_per_pixel,
                                                             const ParameterDictionary &parameters,
                                                             const IntegratorBase *base, const Film *film_prototype,
                                                             GPUMemoryAllocator &allocator) {
    auto integrator = std::make_unique<MLTPathIntegrator>();

    integrator->base = std::make_unique<IntegratorBase>(*base);
    integrator->mutations_per_pixel = mutations_per_pixel;
    integrator->large_step_probability = parameters.get_float("largestepprobability", 0.3);
    integrator->sigma = parameters.get_float("sigma", 0.01);

    integrator->film_dimension = integrator->base->camera->get_camera_base()->resolution;
    integrator->cie_y = std::make_unique<Spectrum>(*parameters.global_spectra->cie_y);

    const long long total_samples =
        static_cast<long long>(mutations_per_pixel) * integrator->film_dimension.x * integrator->film_dimension.y;
    integrator->num_samplers = std::thread::hardware_concurrency();
    integrator->mutations_per_sampler = divide_and_ceil<long long>(total_samples, integrator->num_samplers);

    integrator->max_depth = parameters.get_integer("maxdepth", 5);
    integrator->regularize = parameters.get_bool("regularize", false);

    for (int i = 0; i < integrator->num_samplers; ++i) {
        integrator->chain_films.push_back(film_prototype->create_empty(allocator));
    }

    return integrator;
}

PathSample MLTPathIntegrator::generate_path_sample(Sampler *sampler) const {
    const auto lu = sampler->get_1d();
    auto lambda = SampledWavelengths::sample_visible(lu);

    const auto u = sampler->get_2d();
    const auto p_film = Point2f(u.x * film_dimension.x, u.y * film_dimension.y);
    const auto camera_sample = CameraSample(p_film, 1);

    const auto ray = base->camera->generate_ray(camera_sample, sampler);

    Real depth = 0;
    const auto radiance = ray.weight * MegakernelPathIntegrator::evaluate_Li_volume(&depth, ray.ray, lambda, sampler,
                                                                                    base.get(), max_depth, regularize);

    return PathSample(p_film, radiance, lambda, depth);
}

double MLTPathIntegrator::render(Film *film, GreyScaleFilm &heat_map) const {
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    ThreadPool thread_pool(num_samplers);

    const auto num_paths_per_worker = divide_and_ceil<int>(film_dimension.x * film_dimension.y, num_samplers);

    const auto num_bootstrap_paths = static_cast<size_t>(num_paths_per_worker) * num_samplers;

    if (num_paths_per_worker <= 0) {
        REPORT_FATAL_ERROR();
    }

    std::vector<double> importance_per_path(num_bootstrap_paths);

    thread_pool.parallel_execute(0, num_samplers, [&](const int worker_idx) {
        build_bootstrap_samples(worker_idx, num_paths_per_worker, importance_per_path, this);
    });

    const double sum_importance = std::accumulate(importance_per_path.begin(), importance_per_path.end(), 0.0);

    const double normalization = sum_importance / num_bootstrap_paths;

    const long long total_samples = static_cast<long long>(mutations_per_pixel) * film_dimension.x * film_dimension.y;

    printf("MLT-PATH:\n");
    printf("    number of bootstrap paths: %zu\n", num_bootstrap_paths);
    printf("    MLT samplers: %d\n", num_samplers);
    printf("    mutations per sampler: %lld\n", mutations_per_sampler);
    printf("    normalization: %.6f\n", normalization);

    if (normalization <= 0.0) {
        REPORT_FATAL_ERROR();
    }

    std::vector<GreyScaleFilm> chain_heat_maps;
    for (int i = 0; i < num_samplers; ++i) {
        chain_heat_maps.emplace_back(film_dimension);
    }

    thread_pool.parallel_execute(0, num_samplers, [&](const int worker_idx) {
        mutate_chain(worker_idx, chain_films[worker_idx], chain_heat_maps[worker_idx], this);
    });

    // Account for the extra mutations from rounding every chain up.
    const Real merge_scale = static_cast<Real>(double(total_samples) / (double(mutations_per_sampler) * num_samplers));
    // Merge in worker order on the CPU; Film accumulation is not thread-safe.
    for (int i = 0; i < num_samplers; ++i) {
        film->add_splats(*chain_films[i], merge_scale);
        heat_map.add_samples(chain_heat_maps[i], merge_scale);
    }

    return normalization;
}

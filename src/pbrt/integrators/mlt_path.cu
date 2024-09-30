#include "pbrt/base/camera.h"
#include "pbrt/base/film.h"
#include "pbrt/base/filter.h"
#include "pbrt/base/integrator_base.h"
#include "pbrt/base/sampler.h"
#include "pbrt/films/grey_scale_film.h"
#include "pbrt/integrators/mlt_path.h"
#include "pbrt/integrators/path.h"
#include "pbrt/samplers/mlt.h"
#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/spectrum_util/global_spectra.h"

const size_t NUM_MLT_SAMPLERS = 1024 * 4;
// for unknown reason bugs triggered when NUM_MLT_SAMPLERS set too large (> 1024 * 4)

struct BatchSample {
    PathSample path_sample;
    FloatType weight;

    PBRT_GPU
    BatchSample(FloatType _x, FloatType _y, const SampledSpectrum &_radiance,
                const SampledWavelengths &_lambda, FloatType _weight)
        : path_sample(PathSample(_x, _y, _radiance, _lambda)), weight(_weight) {}
};

__global__ void build_seed_path(PathSample *initial_paths, PathSample *seed_path_candidates,
                                uint num_seed_paths_per_worker, double *sum_illuminance_array,
                                double *luminance_of_paths, MLTPathIntegrator *mlt_integrator,
                                RNG *rngs, const Filter *filter) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= NUM_MLT_SAMPLERS) {
        return;
    }

    sum_illuminance_array[worker_idx] = 0.0;

    auto sampler = &mlt_integrator->samplers[worker_idx];
    auto mlt_sampler = &mlt_integrator->mlt_samplers[worker_idx];

    auto rng = &rngs[worker_idx];

    mlt_sampler->init(rng->uniform<int>());

    mlt_sampler->large_step = 1;

    for (int path_idx = 0; path_idx < num_seed_paths_per_worker; path_idx++) {
        mlt_sampler->init_sample_idx();
        auto path_sample = mlt_integrator->generate_new_path(sampler, filter);
        seed_path_candidates[worker_idx * num_seed_paths_per_worker + path_idx] = path_sample;

        mlt_sampler->global_time += 1;
        mlt_sampler->clear_backup_samples();

        auto illuminance =
            mlt_integrator->compute_luminance(path_sample.radiance, path_sample.lambda);
        luminance_of_paths[worker_idx * num_seed_paths_per_worker + path_idx] = illuminance;

        sum_illuminance_array[worker_idx] += illuminance;
    }

    int selected_path_id = -1;
    const double threshold = rng->uniform<FloatType>() * sum_illuminance_array[worker_idx];
    double accumulated_importance = 0.0;
    for (int path_idx = 0; path_idx < num_seed_paths_per_worker; path_idx++) {
        accumulated_importance +=
            luminance_of_paths[worker_idx * num_seed_paths_per_worker + path_idx];

        if (accumulated_importance >= threshold) {
            selected_path_id = path_idx;
            break;
        }
    }

    if (selected_path_id < 0) {
        REPORT_FATAL_ERROR();
    }

    initial_paths[worker_idx] =
        seed_path_candidates[worker_idx * num_seed_paths_per_worker + selected_path_id];
}

__global__ void wavefront_render(BatchSample *batch_samples, PathSample *path_samples,
                                 MLTPathIntegrator *mlt_integrator, RNG *rngs, const Filter *filter,
                                 double brightness) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= NUM_MLT_SAMPLERS) {
        return;
    }

    auto rng = &rngs[worker_idx];
    const auto p_large = 0.5;
    // TODO: make p_large configurable

    auto sampler = &mlt_integrator->samplers[worker_idx];
    auto mlt_sampler = &mlt_integrator->mlt_samplers[worker_idx];

    auto path = path_samples[worker_idx];

    const auto offset = worker_idx * 2;

    mlt_sampler->init_sample_idx();
    mlt_sampler->large_step = rng->uniform<FloatType>() < p_large;

    auto new_path = mlt_integrator->generate_new_path(sampler, filter);

    const auto new_path_luminance =
        mlt_integrator->compute_luminance(new_path.radiance, new_path.lambda);
    const auto path_luminance = mlt_integrator->compute_luminance(path.radiance, path.lambda);

    double accept_probability = std::min<FloatType>(1.0, new_path_luminance / path_luminance);

    const double new_path_weight = (accept_probability + mlt_sampler->large_step) /
                                   (new_path_luminance / brightness + p_large);
    const double path_weight = (1.0 - accept_probability) / (path_luminance / brightness + p_large);

    batch_samples[offset + 0] =
        BatchSample(path.x, path.y, path.radiance, path.lambda, path_weight);

    batch_samples[offset + 1] =
        BatchSample(new_path.x, new_path.y, new_path.radiance, new_path.lambda, new_path_weight);

    if (rng->uniform<FloatType>() < accept_probability) {
        path = new_path;

        if (mlt_sampler->large_step) {
            mlt_sampler->large_step_time = mlt_sampler->global_time;
        }
        mlt_sampler->global_time++;
        mlt_sampler->clear_backup_samples();
    } else {
        mlt_sampler->recover_samples();
    }

    path_samples[worker_idx] = path;
}

MLTPathIntegrator *MLTPathIntegrator::create(std::optional<int> samples_per_pixel,
                                             const ParameterDictionary &parameters,
                                             const IntegratorBase *base,
                                             std::vector<void *> &gpu_dynamic_pointers) {
    MLTPathIntegrator *integrator;
    CHECK_CUDA_ERROR(cudaMallocManaged(&integrator, sizeof(MLTPathIntegrator)));
    gpu_dynamic_pointers.push_back(integrator);

    if (samples_per_pixel.has_value()) {
        integrator->mutation_per_pixel = samples_per_pixel.value();
    } else {
        integrator->mutation_per_pixel = parameters.get_integer("mutationsperpixel", 10);
    }

    integrator->base = base;

    CHECK_CUDA_ERROR(
        cudaMallocManaged(&(integrator->mlt_samplers), NUM_MLT_SAMPLERS * sizeof(MLTSampler)));
    gpu_dynamic_pointers.push_back(integrator->mlt_samplers);

    CHECK_CUDA_ERROR(
        cudaMallocManaged(&(integrator->samplers), NUM_MLT_SAMPLERS * sizeof(Sampler)));
    gpu_dynamic_pointers.push_back(integrator->samplers);

    for (uint idx = 0; idx < NUM_MLT_SAMPLERS; ++idx) {
        integrator->samplers[idx].init(&(integrator->mlt_samplers[idx]));
    }

    integrator->film_dimension = base->camera->get_camerabase()->resolution;
    integrator->max_depth = 5;
    integrator->cie_y = parameters.global_spectra->cie_y;

    return integrator;
}

PBRT_GPU
PathSample MLTPathIntegrator::generate_new_path(Sampler *sampler, const Filter *filter) const {
    // TODO: move filter into IntegratorBase
    auto pixel_pos = sampler->get_2d();

    auto pixel_x = pixel_pos.x * film_dimension.x;
    auto pixel_y = pixel_pos.y * film_dimension.y;

    auto pixel_coord = Point2i(std::floor(pixel_x), std::floor(pixel_y));
    auto pixel_offset = Point2f(pixel_x - pixel_coord.x, pixel_y - pixel_coord.y);

    auto fs = filter->sample(pixel_offset);

    auto camera_sample = CameraSample(Point2f(pixel_x, pixel_y) + Vector2f(0.5, 0.5), fs.weight);

    auto lu = sampler->get_1d();
    auto lambda = SampledWavelengths::sample_visible(lu);

    auto ray = base->camera->generate_ray(camera_sample, sampler);

    auto radiance = ray.weight * PathIntegrator::eval_li(ray.ray, lambda, base, sampler, max_depth);

    return PathSample(pixel_pos.x, pixel_pos.y, radiance, lambda);
}

void MLTPathIntegrator::render(Film *film, GreyScaleFilm &heat_map, const Filter *filter) {
    const auto num_paths_per_worker = film_dimension.x * film_dimension.y / NUM_MLT_SAMPLERS;
    const auto num_seed_paths = num_paths_per_worker * NUM_MLT_SAMPLERS;

    PathSample *seed_path_candidates;
    PathSample *path_samples;
    double *sum_illuminance_array;
    double *luminance_of_paths;

    CHECK_CUDA_ERROR(cudaMallocManaged(&seed_path_candidates, sizeof(PathSample) * num_seed_paths));
    CHECK_CUDA_ERROR(cudaMallocManaged(&sum_illuminance_array, sizeof(double) * NUM_MLT_SAMPLERS));
    CHECK_CUDA_ERROR(cudaMallocManaged(&luminance_of_paths, sizeof(double) * num_seed_paths));
    CHECK_CUDA_ERROR(cudaMallocManaged(&path_samples, sizeof(PathSample) * NUM_MLT_SAMPLERS));

    RNG *rngs;
    CHECK_CUDA_ERROR(cudaMallocManaged(&rngs, sizeof(RNG) * NUM_MLT_SAMPLERS));
    for (uint idx = 0; idx < NUM_MLT_SAMPLERS; ++idx) {
        rngs[idx].set_sequence(idx);
    }

    auto threads = 32;
    auto blocks = divide_and_ceil<uint>(NUM_MLT_SAMPLERS, threads);
    build_seed_path<<<blocks, threads>>>(path_samples, seed_path_candidates, num_paths_per_worker,
                                         sum_illuminance_array, luminance_of_paths, this, rngs,
                                         filter);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    auto sum_illuminance = 0.0;
    for (uint idx = 0; idx < NUM_MLT_SAMPLERS; ++idx) {
        sum_illuminance += sum_illuminance_array[idx];
    }

    const double brightness = sum_illuminance / num_seed_paths;

    for (auto ptr :
         std::vector<void *>({seed_path_candidates, sum_illuminance_array, luminance_of_paths})) {
        CHECK_CUDA_ERROR(cudaFree(ptr));
    }

    BatchSample *batch_samples;
    CHECK_CUDA_ERROR(cudaMallocManaged(&batch_samples, sizeof(BatchSample) * 2 * NUM_MLT_SAMPLERS));

    auto total_pass =
        (long long)(mutation_per_pixel)*film_dimension.x * film_dimension.y / NUM_MLT_SAMPLERS;

    for (uint pass = 0; pass < total_pass; ++pass) {
        if (pass % 50 == 0 || pass == total_pass - 1) {
            printf("mutation: %u, pass %d/%llu\n", mutation_per_pixel, pass + 1, total_pass);
        }
        
        // TODO: render preview with OpenGL
        wavefront_render<<<blocks, threads>>>(batch_samples, path_samples, this, rngs, filter,
                                              brightness);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        for (uint idx = 0; idx < 2 * NUM_MLT_SAMPLERS; ++idx) {
            auto path_sample = &batch_samples[idx].path_sample;
            auto weight = batch_samples[idx].weight;

            auto pixel_coord =
                Point2i(path_sample->x * film_dimension.x, path_sample->y * film_dimension.y);

            film->add_sample(pixel_coord, path_sample->radiance, path_sample->lambda, weight);
            // TODO: change film->add_sample() to Film::AddSplat() from PBRT-v4

            heat_map.add_sample(pixel_coord, weight);
        }
    }

    for (auto ptr : std::vector<void *>({path_samples, batch_samples, rngs})) {
        CHECK_CUDA_ERROR(cudaFree(ptr));
    }
}

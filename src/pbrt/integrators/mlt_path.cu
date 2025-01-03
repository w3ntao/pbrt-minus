#include "pbrt/base/camera.h"
#include "pbrt/base/film.h"
#include "pbrt/base/integrator_base.h"
#include "pbrt/base/sampler.h"
#include "pbrt/films/grey_scale_film.h"
#include "pbrt/gui/gl_object.h"
#include "pbrt/integrators/mlt_path.h"
#include "pbrt/integrators/path.h"
#include "pbrt/samplers/mlt.h"
#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/spectrum_util/global_spectra.h"

const size_t NUM_MLT_SAMPLERS = 1024 * 256;

struct MLTSample {
    PathSample path_sample;
    FloatType weight;
    FloatType sampling_density;

    PBRT_GPU
    MLTSample(const Point2f p_film, const SampledSpectrum &_radiance,
              const SampledWavelengths &_lambda, FloatType _weight, FloatType _sampling_density)
        : path_sample(PathSample(p_film, _radiance, _lambda)), weight(_weight),
          sampling_density(_sampling_density) {}
};

__global__ void build_seed_path(PathSample *initial_paths, PathSample *seed_path_candidates,
                                uint num_seed_paths_per_worker, double *sum_illuminance_array,
                                double *luminance_of_paths, MLTPathIntegrator *mlt_integrator,
                                RNG *rngs) {
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
        auto path_sample = mlt_integrator->generate_path_sample(sampler);
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

__global__ void wavefront_render(MLTSample *mlt_samples, PathSample *path_samples,
                                 MLTPathIntegrator *mlt_integrator, RNG *rngs, double brightness) {
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

    auto new_path = mlt_integrator->generate_path_sample(sampler);

    const auto new_path_luminance =
        mlt_integrator->compute_luminance(new_path.radiance, new_path.lambda);
    const auto path_luminance = mlt_integrator->compute_luminance(path.radiance, path.lambda);

    FloatType accept_probability = std::min<FloatType>(1.0, new_path_luminance / path_luminance);

    const FloatType new_path_weight = (accept_probability + mlt_sampler->large_step) /
                                      (new_path_luminance / brightness + p_large);
    const FloatType path_weight =
        (1.0 - accept_probability) / (path_luminance / brightness + p_large);

    mlt_samples[offset + 0] =
        MLTSample(path.p_film, path.radiance, path.lambda, path_weight, accept_probability);

    mlt_samples[offset + 1] = MLTSample(new_path.p_film, new_path.radiance, new_path.lambda,
                                        new_path_weight, 1 - accept_probability);

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
        integrator->mutation_per_pixel = parameters.get_integer("mutationsperpixel", 4);
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
    integrator->cie_y = parameters.global_spectra->cie_y;

    integrator->max_depth = parameters.get_integer("maxdepth", 5);
    integrator->regularize = parameters.get_bool("regularize", false);

    return integrator;
}

PBRT_GPU
PathSample MLTPathIntegrator::generate_path_sample(Sampler *sampler) const {
    auto u = sampler->get_2d();

    auto p_film = Point2f(u.x * film_dimension.x, u.y * film_dimension.y);

    auto camera_sample = CameraSample(p_film, 1);

    auto lu = sampler->get_1d();
    auto lambda = SampledWavelengths::sample_visible(lu);

    auto ray = base->camera->generate_ray(camera_sample, sampler);

    auto radiance =
        ray.weight * PathIntegrator::eval_li(ray.ray, lambda, base, sampler, max_depth, regularize);

    return PathSample(p_film, radiance, lambda);
}

void MLTPathIntegrator::render(Film *film, GreyScaleFilm &heat_map, const bool preview) {
    const auto image_resolution = film->get_resolution();
    std::vector<void *> gpu_dynamic_pointers;

    uint8_t *gpu_frame_buffer = nullptr;
    GLObject gl_object;
    if (preview) {
        gl_object.init("initializing", image_resolution);
        CHECK_CUDA_ERROR(cudaMallocManaged(
            &gpu_frame_buffer, sizeof(uint8_t) * 3 * image_resolution.x * image_resolution.y));
        gpu_dynamic_pointers.push_back(gpu_frame_buffer);
    }

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

    constexpr uint threads = 64;
    const uint blocks = divide_and_ceil<uint>(NUM_MLT_SAMPLERS, threads);
    // TODO: stratify seed path building
    build_seed_path<<<blocks, threads>>>(path_samples, seed_path_candidates, num_paths_per_worker,
                                         sum_illuminance_array, luminance_of_paths, this, rngs);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    auto sum_illuminance = 0.0;
    for (uint idx = 0; idx < NUM_MLT_SAMPLERS; ++idx) {
        sum_illuminance += sum_illuminance_array[idx];
    }

    const double brightness = sum_illuminance / num_seed_paths;

    if (brightness <= 0.0) {
        REPORT_FATAL_ERROR();
    }

    for (auto ptr :
         std::vector<void *>({seed_path_candidates, sum_illuminance_array, luminance_of_paths})) {
        CHECK_CUDA_ERROR(cudaFree(ptr));
    }

    MLTSample *mlt_samples;
    CHECK_CUDA_ERROR(cudaMallocManaged(&mlt_samples, sizeof(MLTSample) * 2 * NUM_MLT_SAMPLERS));

    auto total_pass =
        (long long)(mutation_per_pixel)*film_dimension.x * film_dimension.y / NUM_MLT_SAMPLERS;

    if (total_pass == 0) {
        REPORT_FATAL_ERROR();
    }

    for (uint pass = 0; pass < total_pass; ++pass) {
        // TODO: render preview with OpenGL
        wavefront_render<<<blocks, threads>>>(mlt_samples, path_samples, this, rngs, brightness);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        for (uint idx = 0; idx < 2 * NUM_MLT_SAMPLERS; ++idx) {
            auto path_sample = &mlt_samples[idx].path_sample;
            auto p_film = path_sample->p_film;
            auto weight = mlt_samples[idx].weight; // TODO: check if this weight is necessary
            auto sampling_density = mlt_samples[idx].sampling_density;

            auto pixel_x = clamp<int>(std::floor(p_film.x), 0, film_dimension.x - 1);
            auto pixel_y = clamp<int>(std::floor(p_film.y), 0, film_dimension.y - 1);

            auto pixel_coord = Point2i(pixel_x, pixel_y);

            film->add_splat(p_film, path_sample->radiance, path_sample->lambda);

            heat_map.add_sample(pixel_coord, sampling_density);
        }

        if (preview) {
            film->copy_to_frame_buffer(gpu_frame_buffer, 1.0 / mutation_per_pixel);

            gl_object.draw_frame(gpu_frame_buffer,
                                 GLObject::assemble_title(FloatType(pass + 1) / total_pass),
                                 image_resolution);
        }
    }

    for (auto ptr : std::vector<void *>({path_samples, mlt_samples, rngs})) {
        CHECK_CUDA_ERROR(cudaFree(ptr));
    }
}

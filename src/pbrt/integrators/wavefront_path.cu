#include <pbrt/accelerator/hlbvh.h>
#include <pbrt/base/camera.h>
#include <pbrt/base/film.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/light.h>
#include <pbrt/base/material.h>
#include <pbrt/base/sampler.h>
#include <pbrt/gui/gl_helper.h>
#include <pbrt/integrators/megakernel_path.h>
#include <pbrt/integrators/wavefront_path.h>
#include <pbrt/light_samplers/power_light_sampler.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectra/densely_sampled_spectrum.h>
#include <pbrt/util/russian_roulette.h>
#include <pbrt/util/thread_pool.h>
#include <pbrt/util/timer.h>

static constexpr int NUM_THREADS = 256;

struct FrameBuffer {
    int pixel_idx;
    int sample_idx;
    SampledSpectrum radiance;
    SampledWavelengths lambda;
    Real weight;

    // to help sorting
    PBRT_CPU_GPU
    bool operator<(const FrameBuffer &right) const {
        if (pixel_idx < right.pixel_idx) {
            return true;
        }

        if (pixel_idx > right.pixel_idx) {
            return false;
        }

        return sample_idx < right.sample_idx;
    }
};

struct MISParameter {
    bool specular_bounce = false;
    bool any_non_specular_bounces = false;

    pbrt::optional<Real> prev_direction_pdf;
    pbrt::optional<SurfaceInteraction> prev_interaction;

    PBRT_CPU_GPU
    void reset() {
        specular_bounce = false;
        any_non_specular_bounces = false;

        prev_direction_pdf = {};
        prev_interaction = pbrt::optional<SurfaceInteraction>();
    }
};

static __global__ void gpu_reset_path_state(WavefrontPathIntegrator::PathState *path_state) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= WavefrontPathIntegrator::PATH_POOL_SIZE) {
        return;
    }

    path_state->reset_path(worker_idx);
}

__global__ void control_logic(const WavefrontPathIntegrator::PathState *path_state,
                              WavefrontPathIntegrator::Queues *queues, const int ping_pong_idx,
                              const IntegratorBase *base, const int max_depth) {
    const int path_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_idx >= WavefrontPathIntegrator::PATH_POOL_SIZE || path_state->finished[path_idx]) {
        return;
    }

    auto &depth = path_state->depth[path_idx];
    auto &beta = path_state->beta[path_idx];

    const auto ray = path_state->camera_rays[path_idx].ray;
    const auto &lambda = path_state->lambdas[path_idx];

    const auto specular_bounce = path_state->mis_parameters[path_idx].specular_bounce;
    auto &L = path_state->L[path_idx];

    const auto prev_interaction = path_state->mis_parameters[path_idx].prev_interaction;
    const auto prev_direction_pdf = path_state->mis_parameters[path_idx].prev_direction_pdf;
    const auto multi_transmittance_pdf = path_state->multi_transmittance_pdf[path_idx];

    bool should_terminate_path = depth >= max_depth || !beta.is_positive();

    if (!should_terminate_path) {
        auto &next_time_russian_roulette = path_state->next_time_russian_roulette[path_idx];
        if (depth >= next_time_russian_roulette) {
            // possibly terminate the path with Russian roulette
            should_terminate_path |= russian_roulette(beta, &path_state->samplers[path_idx],
                                                      &next_time_russian_roulette);
        }

        if (path_state->in_volume[path_idx]) {
            // start a new ray without evaluating material
            queues->rays->append_path(path_idx);
            return;
        }

        if (!should_terminate_path && !path_state->shape_intersections[path_idx]) {
            // sample infinite lights
            for (int idx = 0; idx < base->infinite_light_num; ++idx) {
                const auto light = base->infinite_lights[idx];
                const auto Le = light->le(ray, lambda);

                if (depth == 0 || specular_bounce) {
                    L += beta * Le;
                } else {
                    // Compute MIS weight for infinite light
                    const Real pdf_light = base->light_sampler->pmf(light) *
                                           light->pdf_li(*prev_interaction, ray.d, true);

                    const Real dir_pdf = *prev_direction_pdf * multi_transmittance_pdf.average();

                    const Real w = power_heuristic(1, dir_pdf, 1, pdf_light);

                    L += beta * w * Le;
                }
            }

            should_terminate_path = true;
        }
    }

    if (should_terminate_path) {
        const int queue_idx = atomicAdd(&queues->ping_pong_counter[ping_pong_idx], 1);
        queues->ping_pong_buffer[ping_pong_idx][queue_idx] = FrameBuffer{
            .pixel_idx = path_state->pixel_indices[path_idx],
            .sample_idx = path_state->sample_indices[path_idx],
            .radiance = L * path_state->camera_rays[path_idx].weight,
            .lambda = lambda,
            .weight = path_state->camera_samples[path_idx].filter_weight,
        };

        queues->new_paths->append_path(path_idx);
        return;
    }

    const auto &surface_interaction = path_state->shape_intersections[path_idx]->interaction;

    if (!surface_interaction.material) {
        // intersecting material-less interface
        depth += IntegratorBase::interface_bounce_contribution;
        queues->interface_material->append_path(path_idx);
        return;
    }

    // otherwise intersecting primitives with material

    const SampledSpectrum Le = surface_interaction.le(-ray.d, lambda);
    if (Le.is_positive()) {
        if (depth == 0 || specular_bounce)
            L += beta * Le;
        else {
            // Compute MIS weight for area light
            const auto area_light = surface_interaction.area_light;
            const Real pdf_light =
                base->light_sampler->pmf(area_light) * area_light->pdf_li(*prev_interaction, ray.d);

            const Real dir_pdf = *prev_direction_pdf * multi_transmittance_pdf.average();

            const Real w = power_heuristic(1, dir_pdf, 1, pdf_light);

            L += beta * w * Le;
        }
    }

    // for active paths: advance one segment
    depth += 1;

    switch (surface_interaction.material->get_material_type()) {
    case Material::Type::conductor: {
        queues->conductor_material->append_path(path_idx);
        return;
    }

    case Material::Type::coated_conductor: {
        queues->coated_conductor_material->append_path(path_idx);
        return;
    }

    case Material::Type::coated_diffuse: {
        queues->coated_diffuse_material->append_path(path_idx);
        return;
    }

    case Material::Type::dielectric: {
        queues->dielectric_material->append_path(path_idx);
        return;
    }

    case Material::Type::diffuse: {
        queues->diffuse_material->append_path(path_idx);
        return;
    }

    case Material::Type::diffuse_transmission: {
        queues->diffuse_transmission_material->append_path(path_idx);
        return;
    }

    case Material::Type::mix: {
        printf("\nyou should not see MixMaterial here\n\n");
        REPORT_FATAL_ERROR();
    }
    }

    REPORT_FATAL_ERROR();
}

__global__ void write_frame_buffer(Film *film, const FrameBuffer *frame_buffer_queue,
                                   const int num) {
    const int queue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_idx >= num) {
        return;
    }

    const auto pixel_idx = frame_buffer_queue[queue_idx].pixel_idx;
    if (queue_idx > 0 && pixel_idx == frame_buffer_queue[queue_idx - 1].pixel_idx) {
        return;
    }

    for (int idx = queue_idx; idx < num && frame_buffer_queue[idx].pixel_idx == pixel_idx; ++idx) {
        // make sure the same pixels are written by the same thread
        const auto &frame_buffer = frame_buffer_queue[idx];
        film->add_sample(frame_buffer.pixel_idx, frame_buffer.radiance, frame_buffer.lambda,
                         frame_buffer.weight);
    }
}

__global__ void fill_new_path_queue(WavefrontPathIntegrator::Queues *queues) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= WavefrontPathIntegrator::PATH_POOL_SIZE) {
        return;
    }

    queues->new_paths->queue_array[worker_idx] = worker_idx;
}

__global__ void gpu_generate_new_path(WavefrontPathIntegrator::PathState *path_state,
                                      WavefrontPathIntegrator::Queues *queues,
                                      const IntegratorBase *base) {
    const int queue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_idx >= queues->new_paths->counter) {
        return;
    }

    const int path_idx = queues->new_paths->queue_array[queue_idx];

    const auto unique_path_id = atomicAdd(&path_state->global_path_counter, 1);
    if (unique_path_id >= path_state->total_path_num) {
        path_state->finished[path_idx] = true;
        return;
    }

    const int width = path_state->image_resolution.x;
    const int height = path_state->image_resolution.y;

    const int pixel_idx = unique_path_id % (width * height);
    const int sample_idx = unique_path_id / (width * height);

    auto sampler = &path_state->samplers[path_idx];

    const auto p_pixel = Point2i(pixel_idx % width, pixel_idx / width);

    sampler->start_pixel_sample(pixel_idx, sample_idx, 0);

    path_state->camera_samples[path_idx] = sampler->get_camera_sample(p_pixel, base->filter);
    const auto lu = sampler->get_1d();
    path_state->lambdas[path_idx] = SampledWavelengths::sample_visible(lu);

    path_state->camera_rays[path_idx] =
        base->camera->generate_ray(path_state->camera_samples[path_idx], sampler);

    path_state->pixel_indices[path_idx] = pixel_idx;
    path_state->sample_indices[path_idx] = sample_idx;
    path_state->depth[path_idx] = 0;

    path_state->reset_path(path_idx);

    queues->rays->append_path(path_idx);
}

void generate_new_path(WavefrontPathIntegrator::PathState *path_state,
                       WavefrontPathIntegrator::Queues *queues, const IntegratorBase *base) {
    if (queues->new_paths->counter <= 0) {
        return;
    }

    gpu_generate_new_path<<<divide_and_ceil(queues->new_paths->counter, NUM_THREADS),
                            NUM_THREADS>>>(path_state, queues, base);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    queues->new_paths->counter = 0;
}

__global__ void
gpu_evaluate_interface_material(const WavefrontPathIntegrator::Queues::SingleQueue *queue,
                                const WavefrontPathIntegrator *integrator) {
    const int queue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_idx >= queue->counter) {
        return;
    }

    const auto path_state = &integrator->path_state;
    const int path_idx = queue->queue_array[queue_idx];

    const auto &surface_interaction = path_state->shape_intersections[path_idx]->interaction;

    auto &ray = path_state->camera_rays[path_idx].ray;
    ray = surface_interaction.spawn_ray(ray.d);

    integrator->queues.rays->append_path(path_idx);
}

__global__ void
gpu_evaluate_material(const WavefrontPathIntegrator::Queues::SingleQueue *material_queue,
                      const WavefrontPathIntegrator *integrator) {
    const int queue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_idx >= material_queue->counter) {
        return;
    }

    auto &path_state = integrator->path_state;

    const int path_idx = material_queue->queue_array[queue_idx];

    auto &lambda = path_state.lambdas[path_idx];

    auto &sampler = path_state.samplers[path_idx];

    auto &surface_interaction = path_state.shape_intersections[path_idx]->interaction;

    path_state.bsdf[path_idx] = surface_interaction.get_bsdf(lambda, integrator->base->camera,
                                                             sampler.get_samples_per_pixel());

    integrator->sample_bsdf(path_idx, &path_state);

    integrator->queues.rays->append_path(path_idx);
}

__global__ void ray_cast(WavefrontPathIntegrator::PathState *path_state,
                         WavefrontPathIntegrator::Queues *queues, const IntegratorBase *base) {
    const int ray_queue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_queue_idx >= queues->rays->counter) {
        return;
    }

    const int path_idx = queues->rays->queue_array[ray_queue_idx];
    const auto &ray = path_state->camera_rays[path_idx].ray;

    const auto optional_intersection = base->intersect(ray, Infinity);
    path_state->shape_intersections[path_idx] = optional_intersection;

    path_state->in_volume[path_idx] = false;

    if (ray.medium) {
        queues->volume->append_path(path_idx);
    }
}

__global__ void volume_scatter(WavefrontPathIntegrator::PathState *path_state,
                               WavefrontPathIntegrator::Queues *queues,
                               const IntegratorBase *base) {

    const int volume_queue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (volume_queue_idx >= queues->volume->counter) {
        return;
    }

    const int path_idx = queues->volume->queue_array[volume_queue_idx];
    auto &ray = path_state->camera_rays[path_idx].ray;

    const auto optional_intersection = path_state->shape_intersections[path_idx];

    const auto &lambda = path_state->lambdas[path_idx];

    auto sampler = &path_state->samplers[path_idx];
    auto &beta = path_state->beta[path_idx];
    auto &L = path_state->L[path_idx];
    auto &multi_transmittance_pdf = path_state->multi_transmittance_pdf[path_idx];

    const SampledSpectrum sigma_a = ray.medium->sigma_a->sample(lambda);
    const SampledSpectrum sigma_s = ray.medium->sigma_s->sample(lambda);
    const SampledSpectrum sigma_t = sigma_a + sigma_s;

    const auto t_max = optional_intersection ? optional_intersection->t_hit : Infinity;

    const auto t = sample_exponential(sampler->get_1d(), sigma_t.average());
    if (t >= t_max) {
        // pass through medium
        multi_transmittance_pdf *= SampledSpectrum::exp(-sigma_t * t_max);
        return;
    }

    // otherwise scatter in medium
    ray.o = ray.at(t);
    beta *= sigma_s / sigma_t;

    SurfaceInteraction medium_interaction;
    medium_interaction.pi = ray.o;
    medium_interaction.wo = -ray.d;
    medium_interaction.medium = ray.medium;

    SampledSpectrum Ld = MegakernelPathIntegrator::sample_Ld_volume(medium_interaction, nullptr,
                                                                    lambda, sampler, base);
    L += beta * Ld;

    auto phase_sample = ray.medium->phase.sample(-ray.d, sampler->get_2d());
    if (!phase_sample) {
        beta = 0;
        return;
    }

    beta *= phase_sample->rho / phase_sample->pdf;

    path_state->mis_parameters[path_idx].prev_direction_pdf = phase_sample->pdf;
    path_state->mis_parameters[path_idx].prev_interaction = medium_interaction;
    ray.d = phase_sample->wi;

    multi_transmittance_pdf = 1;

    path_state->depth[path_idx] += 1;
    path_state->in_volume[path_idx] = true;
    // dont't add this path into ray_cast() immediately
    // that would bypass russian roulette, leading to possibly infinite bounce

    path_state->mis_parameters[path_idx].specular_bounce = false;
    path_state->mis_parameters[path_idx].any_non_specular_bounces = true;
}

PBRT_CPU_GPU
void WavefrontPathIntegrator::sample_bsdf(const int path_idx, const PathState *path_state) const {
    auto &isect = path_state->shape_intersections[path_idx]->interaction;
    auto &lambda = path_state->lambdas[path_idx];

    auto &ray = path_state->camera_rays[path_idx].ray;
    auto sampler = &path_state->samplers[path_idx];

    if (regularize && path_state->mis_parameters[path_idx].any_non_specular_bounces) {
        path_state->bsdf[path_idx].regularize();
    }

    if (pbrt::is_non_specular(path_state->bsdf[path_idx].flags())) {
        const SampledSpectrum Ld = MegakernelPathIntegrator::sample_Ld_volume(
            isect, &path_state->bsdf[path_idx], lambda, sampler, base);
        path_state->L[path_idx] += path_state->beta[path_idx] * Ld;
    }

    // Sample BSDF to get new path direction
    Vector3f wo = -ray.d;
    Real u = sampler->get_1d();
    auto bs = path_state->bsdf[path_idx].sample_f(wo, u, sampler->get_2d());
    if (!bs) {
        path_state->beta[path_idx] = 0;
        return;
    }

    path_state->beta[path_idx] *= bs->f * bs->wi.abs_dot(isect.shading.n.to_vector3()) / bs->pdf;
    path_state->multi_transmittance_pdf[path_idx] = 1;

    path_state->mis_parameters[path_idx].prev_direction_pdf =
        bs->pdf_is_proportional ? path_state->bsdf[path_idx].pdf(wo, bs->wi) : bs->pdf;
    path_state->mis_parameters[path_idx].specular_bounce = bs->is_specular();
    path_state->mis_parameters[path_idx].any_non_specular_bounces |= !bs->is_specular();
    path_state->mis_parameters[path_idx].prev_interaction = isect;

    path_state->camera_rays[path_idx].ray = isect.spawn_ray(bs->wi);
}

void WavefrontPathIntegrator::evaluate_interface_material() {
    if (queues.interface_material->counter <= 0) {
        return;
    }

    const auto blocks = divide_and_ceil(queues.interface_material->counter, MAX_THREADS_PER_BLOCKS);
    gpu_evaluate_interface_material<<<blocks, MAX_THREADS_PER_BLOCKS>>>(queues.interface_material,
                                                                        this);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    queues.interface_material->counter = 0;
}

void WavefrontPathIntegrator::evaluate_material(const Material::Type material_type) {
    const auto material_queue = queues.get_material_queue(material_type);
    if (material_queue->counter <= 0) {
        return;
    }

    constexpr int threads = 256;
    const auto blocks = divide_and_ceil(material_queue->counter, threads);

    gpu_evaluate_material<<<blocks, threads>>>(material_queue, this);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    material_queue->counter = 0;
}

PBRT_CPU_GPU
void WavefrontPathIntegrator::PathState::reset_path(const int path_idx) {
    finished[path_idx] = false;
    shape_intersections[path_idx] = {};

    L[path_idx] = 0;
    beta[path_idx] = 1;
    multi_transmittance_pdf[path_idx] = 1;
    in_volume[path_idx] = false;
    depth[path_idx] = 0;
    next_time_russian_roulette[path_idx] = start_russian_roulette;
    mis_parameters[path_idx].reset();
}

WavefrontPathIntegrator::PathState::PathState(const Point2i &_resolution,
                                              const int _samples_per_pixel,
                                              const std::string &sampler_type)
    : image_resolution(_resolution), samples_per_pixel(_samples_per_pixel),
      sampler_type(sampler_type), total_path_num(static_cast<long long int>(_samples_per_pixel) *
                                                 image_resolution.x * image_resolution.y) {
    global_path_counter = 0;
}

void WavefrontPathIntegrator::PathState::build_path(GPUMemoryAllocator &allocator) {
    if (total_path_num / samples_per_pixel != image_resolution.x * image_resolution.y) {
        REPORT_FATAL_ERROR();
    }

    camera_samples = allocator.allocate<CameraSample>(PATH_POOL_SIZE);
    camera_rays = allocator.allocate<CameraRay>(PATH_POOL_SIZE);
    lambdas = allocator.allocate<SampledWavelengths>(PATH_POOL_SIZE);

    L = allocator.allocate<SampledSpectrum>(PATH_POOL_SIZE);
    beta = allocator.allocate<SampledSpectrum>(PATH_POOL_SIZE);
    multi_transmittance_pdf = allocator.allocate<SampledSpectrum>(PATH_POOL_SIZE);
    shape_intersections = allocator.allocate<pbrt::optional<ShapeIntersection>>(PATH_POOL_SIZE);
    in_volume = allocator.allocate<bool>(PATH_POOL_SIZE);

    depth = allocator.allocate<Real>(PATH_POOL_SIZE);
    next_time_russian_roulette = allocator.allocate<int>(PATH_POOL_SIZE);
    finished = allocator.allocate<bool>(PATH_POOL_SIZE);
    pixel_indices = allocator.allocate<int>(PATH_POOL_SIZE);
    sample_indices = allocator.allocate<int>(PATH_POOL_SIZE);

    bsdf = allocator.allocate<BSDF>(PATH_POOL_SIZE);
    mis_parameters = allocator.allocate<MISParameter>(PATH_POOL_SIZE);
    samplers = Sampler::create_samplers(sampler_type, samples_per_pixel, PATH_POOL_SIZE, allocator);

    gpu_reset_path_state<<<PATH_POOL_SIZE, MAX_THREADS_PER_BLOCKS>>>(this);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

WavefrontPathIntegrator::Queues::Queues(GPUMemoryAllocator &allocator) {
    auto create_new_queue = [&allocator] { return allocator.create<SingleQueue>(allocator); };

    new_paths = create_new_queue();
    rays = create_new_queue();

    volume = create_new_queue();
    interface_material = create_new_queue();
    conductor_material = create_new_queue();
    coated_conductor_material = create_new_queue();
    coated_diffuse_material = create_new_queue();
    dielectric_material = create_new_queue();
    diffuse_material = create_new_queue();
    diffuse_transmission_material = create_new_queue();

    for (const auto ping_pong_idx : {0, 1}) {
        ping_pong_counter[ping_pong_idx] = 0;
        ping_pong_buffer[ping_pong_idx] = allocator.allocate<FrameBuffer>(PATH_POOL_SIZE);
    }
}

WavefrontPathIntegrator::WavefrontPathIntegrator(const int _samples_per_pixel,
                                                 const std::string &sampler_type,
                                                 const ParameterDictionary &parameters,
                                                 const IntegratorBase *_base,
                                                 GPUMemoryAllocator &allocator)
    : path_state(PathState(_base->camera->get_camera_base()->resolution, _samples_per_pixel,
                           sampler_type)),
      queues(allocator), base(_base), samples_per_pixel(_samples_per_pixel),
      max_depth(parameters.get_integer("maxdepth", 5)),
      regularize(parameters.get_bool("regularize", false)) {
    path_state.build_path(allocator);
}

void WavefrontPathIntegrator::render(Film *film, const bool preview) {
    printf("wavefront pathtracing: path pool size: %u\n", PATH_POOL_SIZE);

    const auto image_resolution = this->path_state.image_resolution;
    const auto num_pixels = image_resolution.x * image_resolution.y;

    GLHelper gl_helper;
    if (preview) {
        gl_helper.init("initializing", image_resolution);
    }

    // generate new paths for the whole pool
    fill_new_path_queue<<<PATH_POOL_SIZE, NUM_THREADS>>>(&queues);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    queues.new_paths->counter = PATH_POOL_SIZE;
    queues.rays->counter = 0;

    generate_new_path(&path_state, &queues, base);

    if (queues.rays->counter <= 0) {
        REPORT_FATAL_ERROR();
    }

    auto timer = Timer();
    std::mutex mutex[2];
    auto clear_ping_pong_buffer = [&](const int ping_pong_idx) -> bool {
        const int size = queues.ping_pong_counter[ping_pong_idx];
        if (size <= 0) {
            return false;
        }

        write_frame_buffer<<<divide_and_ceil(size, NUM_THREADS), NUM_THREADS>>>(
            film, queues.ping_pong_buffer[ping_pong_idx], size);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        queues.ping_pong_counter[ping_pong_idx] = 0;

        return true;
    };

    ThreadPool thread_pool;
    int pass = 0;
    while (queues.rays->counter > 0) {
        const int ping_pong_idx = pass % 2;

        if (DEBUG_MODE) {
            printf("pass: %d, ray_cast(): %d\n", pass, queues.rays->counter);
        }
        ray_cast<<<divide_and_ceil(queues.rays->counter, NUM_THREADS), NUM_THREADS>>>(
            &path_state, &queues, base);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        queues.rays->counter = 0;

        timer.stop("stage 0 ray_cast()");

        if (queues.volume->counter > 0) {
            volume_scatter<<<divide_and_ceil(queues.volume->counter, NUM_THREADS), NUM_THREADS>>>(
                &path_state, &queues, base);
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            queues.volume->counter = 0;

            timer.stop("stage 1 volume_scatter()");
        }

        mutex[ping_pong_idx].lock();
        if (clear_ping_pong_buffer(ping_pong_idx) && preview) {
            film->copy_to_frame_buffer(gl_helper.gpu_frame_buffer);

            const auto current_sample_idx =
                std::min<int>(path_state.global_path_counter / num_pixels, samples_per_pixel);

            gl_helper.draw_frame(
                GLHelper::assemble_title(Real(current_sample_idx) / samples_per_pixel));
        }
        timer.stop("stage 2 write_frame_buffer()");

        if (queues.ping_pong_counter[ping_pong_idx] != 0) {
            REPORT_FATAL_ERROR();
        }

        control_logic<<<divide_and_ceil(PATH_POOL_SIZE, NUM_THREADS), NUM_THREADS>>>(
            &path_state, &queues, ping_pong_idx, base, max_depth);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        mutex[ping_pong_idx].unlock();

        timer.stop("stage 3 control_logic()");

        if (queues.ping_pong_counter[ping_pong_idx] > 0) {
            mutex[ping_pong_idx].lock();
            thread_pool.submit([&, idx = ping_pong_idx] {
                const auto buffer_queue = queues.ping_pong_buffer[idx];
                const auto size = queues.ping_pong_counter[idx];
                std::sort(buffer_queue, buffer_queue + size, std::less{});
                mutex[idx].unlock();
            });
        }

        if (DEBUG_MODE) {
            printf("pass: %d, generate_new_path(): %d\n", pass, queues.rays->counter);
        }

        generate_new_path(&path_state, &queues, base);

        timer.stop("stage 4 generate_new_path()");

        if (DEBUG_MODE) {
            printf("pass: %d, evaluate_interface_material(): %d\n", pass,
                   queues.interface_material->counter);
        }

        evaluate_interface_material();
        for (const auto material_type : Material::get_basic_material_types()) {
            if (DEBUG_MODE) {
                printf("pass: %d, evaluate_material(%d): %d\n", pass, material_type,
                       queues.get_material_queue(material_type)->counter);
            }
            evaluate_material(material_type);
        }

        timer.stop("stage 5 evaluate_material()");

        pass += 1;
    }

    thread_pool.sync(); // this is not technically necessary, yet a good practice
    for (auto ping_pong_idx : {0, 1}) {
        mutex[ping_pong_idx].lock();
        clear_ping_pong_buffer(ping_pong_idx);
        mutex[ping_pong_idx].unlock();
    }
    timer.stop("stage 2 write_frame_buffer()");

    printf("wavefront pathtracing benchmark:\n");
    timer.print();
    printf("\n");
}

#include <pbrt/accelerator/hlbvh.h>
#include <pbrt/base/film.h>
#include <pbrt/base/integrator_base.h>
#include <pbrt/base/light.h>
#include <pbrt/base/material.h>
#include <pbrt/base/sampler.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/gui/gl_helper.h>
#include <pbrt/integrators/wavefront_path.h>
#include <pbrt/light_samplers/power_light_sampler.h>
#include <pbrt/samplers/independent.h>
#include <pbrt/samplers/stratified.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/sampled_spectrum.h>
#include <pbrt/spectrum_util/sampled_wavelengths.h>
#include <pbrt/util/math.h>

constexpr uint PATH_POOL_SIZE = 2 * 1024 * 1024;

struct FrameBuffer {
    uint pixel_idx;
    uint sample_idx;
    SampledSpectrum radiance;
    SampledWavelengths lambda;
    FloatType weight;

    // to help sorting
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
    bool specular_bounce = true;
    bool any_non_specular_bounces = false;

    FloatType pdf_bsdf;
    FloatType eta_scale;
    LightSampleContext prev_interaction_light_sample_ctx;

    PBRT_CPU_GPU
    void init() {
        specular_bounce = true;
        any_non_specular_bounces = false;

        pdf_bsdf = NAN;
        eta_scale = 1.0;
    }
};

static __global__ void gpu_init_independent_samplers(Sampler *samplers,
                                                     IndependentSampler *independent_samplers,
                                                     uint num) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num) {
        return;
    }

    samplers[worker_idx].init(&independent_samplers[worker_idx]);
}

static __global__ void gpu_init_stratified_samplers(Sampler *samplers,
                                                    StratifiedSampler *stratified_samplers,
                                                    uint samples_per_dimension, uint num) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num) {
        return;
    }

    stratified_samplers[worker_idx].init(samples_per_dimension);

    samplers[worker_idx].init(&stratified_samplers[worker_idx]);
}

static __global__ void gpu_init_path_state(WavefrontPathIntegrator::PathState *path_state) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= PATH_POOL_SIZE) {
        return;
    }

    path_state->init_new_path(worker_idx);
}

__global__ void control_logic(WavefrontPathIntegrator::PathState *path_state,
                              WavefrontPathIntegrator::Queues *queues, const uint max_depth,
                              const IntegratorBase *base) {
    const uint path_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_idx >= PATH_POOL_SIZE || path_state->finished[path_idx]) {
        return;
    }

    auto &isect = path_state->shape_intersections[path_idx]->interaction;
    const auto ray = path_state->camera_rays[path_idx].ray;
    auto &lambda = path_state->lambdas[path_idx];

    const auto path_length = path_state->path_length[path_idx];
    const auto specular_bounce = path_state->mis_parameters[path_idx].specular_bounce;
    auto &beta = path_state->beta[path_idx];
    auto &L = path_state->L[path_idx];

    const auto prev_interaction_light_sample_ctx =
        path_state->mis_parameters[path_idx].prev_interaction_light_sample_ctx;
    const auto pdf_bsdf = path_state->mis_parameters[path_idx].pdf_bsdf;

    bool should_terminate_path = !path_state->shape_intersections[path_idx].has_value() ||
                                 path_length > max_depth || !beta.is_positive();

    if (!should_terminate_path && path_length > 8) {
        // possibly terminate the path with Russian roulette

        auto &eta_scale = path_state->mis_parameters[path_idx].eta_scale;
        auto &sampler = path_state->samplers[path_idx];
        const auto u = sampler.get_1d();
        // consume this random value anyway to keep samples aligned

        SampledSpectrum russian_roulette_beta = beta * eta_scale;
        if (russian_roulette_beta.max_component_value() < 1) {
            auto q = clamp<FloatType>(1 - russian_roulette_beta.max_component_value(), 0, 0.95);
            if (u < q) {
                beta = SampledSpectrum(0.0);
                should_terminate_path = true;
            } else {
                beta /= 1 - q;
            }
        }
    }

    if (should_terminate_path) {
        if (beta.is_positive()) {
            // sample infinite lights
            for (uint idx = 0; idx < base->infinite_light_num; ++idx) {
                auto light = base->infinite_lights[idx];
                auto Le = light->le(ray, lambda);

                if (path_length == 0 || specular_bounce) {
                    L += beta * Le;
                } else {
                    // Compute MIS weight for infinite light
                    FloatType pdf_light =
                        base->light_sampler->pmf(prev_interaction_light_sample_ctx, light) *
                        light->pdf_li(prev_interaction_light_sample_ctx, ray.d, true);
                    FloatType weight_bsdf = power_heuristic(1, pdf_bsdf, 1, pdf_light);

                    L += beta * weight_bsdf * Le;
                }
            }
        }

        const uint queue_idx = atomicAdd(&queues->frame_buffer_counter, 1);
        queues->frame_buffer_queue[queue_idx] = FrameBuffer{
            .pixel_idx = path_state->pixel_indices[path_idx],
            .sample_idx = path_state->sample_indices[path_idx],
            .radiance = L * path_state->camera_rays[path_idx].weight,
            .lambda = lambda,
            .weight = path_state->camera_samples[path_idx].filter_weight,
        };

        queues->new_paths->append_path(path_idx);
        return;
    }

    SampledSpectrum Le = isect.le(-ray.d, lambda);
    if (Le.is_positive()) {
        if (path_length == 0 || specular_bounce)
            path_state->L[path_idx] += beta * Le;
        else {
            // Compute MIS weight for area light
            auto area_light = isect.area_light;

            FloatType pdf_light =
                base->light_sampler->pmf(prev_interaction_light_sample_ctx, area_light) *
                area_light->pdf_li(prev_interaction_light_sample_ctx, ray.d);
            FloatType weight_light = power_heuristic(1, pdf_bsdf, 1, pdf_light);

            path_state->L[path_idx] += beta * weight_light * Le;
        }
    }

    // for active paths: advance one segment

    path_state->path_length[path_idx] += 1;

    switch (isect.material->get_material_type()) {

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

__global__ void write_frame_buffer(Film *film, WavefrontPathIntegrator::Queues *queues) {
    const uint queue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_idx >= queues->frame_buffer_counter) {
        return;
    }

    const auto pixel_idx = queues->frame_buffer_queue[queue_idx].pixel_idx;
    if (queue_idx > 0 && pixel_idx == queues->frame_buffer_queue[queue_idx - 1].pixel_idx) {
        return;
    }

    for (uint idx = queue_idx; idx < queues->frame_buffer_counter &&
                               queues->frame_buffer_queue[idx].pixel_idx == pixel_idx;
         ++idx) {
        // make sure the same pixels are written by the same thread
        const auto &frame_buffer = queues->frame_buffer_queue[idx];
        film->add_sample(frame_buffer.pixel_idx, frame_buffer.radiance, frame_buffer.lambda,
                         frame_buffer.weight);
    }
}

__global__ void fill_new_path_queue(WavefrontPathIntegrator::Queues *queues) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= PATH_POOL_SIZE) {
        return;
    }

    queues->new_paths->queue_array[worker_idx] = worker_idx;
}

__global__ void generate_new_path(WavefrontPathIntegrator::PathState *path_state,
                                  WavefrontPathIntegrator::Queues *queues,
                                  const IntegratorBase *base) {
    const uint queue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_idx >= queues->new_paths->counter) {
        return;
    }

    const uint path_idx = queues->new_paths->queue_array[queue_idx];

    const auto unique_path_id = atomicAdd(&path_state->global_path_counter, 1);
    if (unique_path_id >= path_state->total_path_num) {
        path_state->finished[path_idx] = true;
        return;
    }

    const uint width = path_state->image_resolution.x;
    const uint height = path_state->image_resolution.y;

    const uint pixel_idx = unique_path_id % (width * height);
    const uint sample_idx = unique_path_id / (width * height);

    auto sampler = &path_state->samplers[path_idx];

    auto p_pixel = Point2i(pixel_idx % width, pixel_idx / width);

    sampler->start_pixel_sample(pixel_idx, sample_idx, 0);

    path_state->camera_samples[path_idx] = sampler->get_camera_sample(p_pixel, base->filter);
    auto lu = sampler->get_1d();
    path_state->lambdas[path_idx] = SampledWavelengths::sample_visible(lu);

    path_state->camera_rays[path_idx] =
        base->camera->generate_ray(path_state->camera_samples[path_idx], sampler);

    path_state->pixel_indices[path_idx] = pixel_idx;
    path_state->sample_indices[path_idx] = sample_idx;
    path_state->path_length[path_idx] = 0;

    path_state->init_new_path(path_idx);

    queues->rays->append_path(path_idx);
}

__global__ void gpu_evaluate_material(WavefrontPathIntegrator::Queues::SingleQueue *material_queue,
                                      WavefrontPathIntegrator *integrator) {
    const uint queue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_idx >= material_queue->counter) {
        return;
    }

    auto path_state = &integrator->path_state;

    const uint path_idx = material_queue->queue_array[queue_idx];

    auto &lambda = path_state->lambdas[path_idx];

    auto sampler = &path_state->samplers[path_idx];

    auto &isect = path_state->shape_intersections[path_idx]->interaction;

    path_state->bsdf[path_idx] =
        isect.get_bsdf(lambda, integrator->base->camera, sampler->get_samples_per_pixel());

    integrator->sample_bsdf(path_idx, path_state);

    integrator->queues.rays->append_path(path_idx);
}

__global__ void ray_cast(WavefrontPathIntegrator::PathState *path_state,
                         WavefrontPathIntegrator::Queues *queues, const IntegratorBase *base) {
    const uint ray_queue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_queue_idx >= queues->rays->counter) {
        return;
    }

    const uint path_idx = queues->rays->queue_array[ray_queue_idx];

    const auto camera_ray = path_state->camera_rays[path_idx];

    path_state->shape_intersections[path_idx] = base->intersect(camera_ray.ray, Infinity);
}

PBRT_CPU_GPU
void WavefrontPathIntegrator::sample_bsdf(uint path_idx, PathState *path_state) const {
    auto &isect = path_state->shape_intersections[path_idx]->interaction;
    auto &lambda = path_state->lambdas[path_idx];

    auto &ray = path_state->camera_rays[path_idx].ray;
    auto sampler = &path_state->samplers[path_idx];

    if (regularize && path_state->mis_parameters[path_idx].any_non_specular_bounces) {
        path_state->bsdf[path_idx].regularize();
    }

    if (pbrt::is_non_specular(path_state->bsdf[path_idx].flags())) {
        SampledSpectrum Ld = sample_ld(isect, &path_state->bsdf[path_idx], lambda, sampler);
        path_state->L[path_idx] += path_state->beta[path_idx] * Ld;
    }

    // Sample BSDF to get new path direction
    Vector3f wo = -ray.d;
    FloatType u = sampler->get_1d();
    auto bs = path_state->bsdf[path_idx].sample_f(wo, u, sampler->get_2d());
    if (!bs) {
        path_state->beta[path_idx] = SampledSpectrum(0.0);
        return;
    }

    path_state->beta[path_idx] *= bs->f * bs->wi.abs_dot(isect.shading.n.to_vector3()) / bs->pdf;

    path_state->mis_parameters[path_idx].pdf_bsdf =
        bs->pdf_is_proportional ? path_state->bsdf[path_idx].pdf(wo, bs->wi) : bs->pdf;
    path_state->mis_parameters[path_idx].specular_bounce = bs->is_specular();
    path_state->mis_parameters[path_idx].any_non_specular_bounces |= (!bs->is_specular());

    if (bs->is_transmission()) {
        path_state->mis_parameters[path_idx].eta_scale *= sqr(bs->eta);
    }

    path_state->mis_parameters[path_idx].prev_interaction_light_sample_ctx = isect;

    path_state->camera_rays[path_idx].ray = isect.spawn_ray(bs->wi);
}

void WavefrontPathIntegrator::evaluate_material(const Material::Type material_type) {
    auto material_queue = this->queues.get_material_queue(material_type);
    if (material_queue->counter <= 0) {
        return;
    }

    constexpr uint threads = 256;
    const auto blocks = divide_and_ceil(material_queue->counter, threads);

    gpu_evaluate_material<<<blocks, threads>>>(material_queue, this);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

PBRT_CPU_GPU
void WavefrontPathIntegrator::PathState::init_new_path(uint path_idx) {
    finished[path_idx] = false;
    shape_intersections[path_idx].reset();

    L[path_idx] = SampledSpectrum(0.0);
    beta[path_idx] = SampledSpectrum(1.0);
    path_length[path_idx] = 0;

    mis_parameters[path_idx].init();
}

void WavefrontPathIntegrator::PathState::create(uint samples_per_pixel, const Point2i &_resolution,
                                                const std::string &sampler_type,
                                                GPUMemoryAllocator &allocator) {
    image_resolution = _resolution;
    global_path_counter = 0;
    total_path_num = samples_per_pixel * image_resolution.x * image_resolution.y;

    camera_samples = allocator.allocate<CameraSample>(PATH_POOL_SIZE);
    camera_rays = allocator.allocate<CameraRay>(PATH_POOL_SIZE);
    lambdas = allocator.allocate<SampledWavelengths>(PATH_POOL_SIZE);

    L = allocator.allocate<SampledSpectrum>(PATH_POOL_SIZE);
    beta = allocator.allocate<SampledSpectrum>(PATH_POOL_SIZE);
    shape_intersections = allocator.allocate<pbrt::optional<ShapeIntersection>>(PATH_POOL_SIZE);

    path_length = allocator.allocate<uint>(PATH_POOL_SIZE);
    finished = allocator.allocate<bool>(PATH_POOL_SIZE);
    pixel_indices = allocator.allocate<uint>(PATH_POOL_SIZE);
    sample_indices = allocator.allocate<uint>(PATH_POOL_SIZE);

    bsdf = allocator.allocate<BSDF>(PATH_POOL_SIZE);
    mis_parameters = allocator.allocate<MISParameter>(PATH_POOL_SIZE);
    samplers = allocator.allocate<Sampler>(PATH_POOL_SIZE);

    constexpr uint threads = 1024;
    uint blocks = divide_and_ceil<uint>(PATH_POOL_SIZE, threads);

    if (sampler_type == "stratified") {
        const auto samples_per_dimension = static_cast<int>(std::sqrt(samples_per_pixel));
        if (samples_per_dimension * samples_per_dimension != samples_per_pixel) {
            REPORT_FATAL_ERROR();
        }

        auto stratified_samplers = allocator.allocate<StratifiedSampler>(PATH_POOL_SIZE);

        gpu_init_stratified_samplers<<<blocks, threads>>>(samplers, stratified_samplers,
                                                          samples_per_dimension, PATH_POOL_SIZE);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    } else if (sampler_type == "independent") {
        auto independent_samplers = allocator.allocate<IndependentSampler>(PATH_POOL_SIZE);

        gpu_init_independent_samplers<<<blocks, threads>>>(samplers, independent_samplers,
                                                           PATH_POOL_SIZE);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    } else {
        REPORT_FATAL_ERROR();
    }

    gpu_init_path_state<<<PATH_POOL_SIZE, threads>>>(this);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void WavefrontPathIntegrator::Queues::init(GPUMemoryAllocator &allocator) {
    new_paths = build_new_queue(allocator);
    rays = build_new_queue(allocator);
    conductor_material = build_new_queue(allocator);
    coated_conductor_material = build_new_queue(allocator);
    coated_diffuse_material = build_new_queue(allocator);
    dielectric_material = build_new_queue(allocator);
    diffuse_material = build_new_queue(allocator);
    diffuse_transmission_material = build_new_queue(allocator);

    frame_buffer_counter = 0;
    frame_buffer_queue = allocator.allocate<FrameBuffer>(PATH_POOL_SIZE);
}

WavefrontPathIntegrator::Queues::SingleQueue *
WavefrontPathIntegrator::Queues::build_new_queue(GPUMemoryAllocator &allocator) {
    auto queue = allocator.allocate<SingleQueue>(PATH_POOL_SIZE);
    queue->counter = 0;
    queue->queue_array = allocator.allocate<uint>(PATH_POOL_SIZE);

    return queue;
}

WavefrontPathIntegrator *WavefrontPathIntegrator::create(uint samples_per_pixel,
                                                         const std::string &sampler_type,
                                                         const ParameterDictionary &parameters,
                                                         const IntegratorBase *base,
                                                         GPUMemoryAllocator &allocator) {
    auto integrator = allocator.allocate<WavefrontPathIntegrator>();

    integrator->samples_per_pixel = samples_per_pixel;

    integrator->base = base;
    integrator->path_state.create(samples_per_pixel, base->camera->get_camerabase()->resolution,
                                  sampler_type, allocator);

    integrator->queues.init(allocator);

    integrator->max_depth = parameters.get_integer("maxdepth", 5);
    integrator->regularize = parameters.get_bool("regularize", false);

    return integrator;
}

PBRT_CPU_GPU
SampledSpectrum WavefrontPathIntegrator::sample_ld(const SurfaceInteraction &intr, const BSDF *bsdf,
                                                   SampledWavelengths &lambda,
                                                   Sampler *sampler) const {
    // Initialize _LightSampleContext_ for light sampling
    LightSampleContext ctx(intr);
    // Try to nudge the light sampling position to correct side of the surface
    BxDFFlags flags = bsdf->flags();
    if (pbrt::is_reflective(flags) && !pbrt::is_transmissive(flags)) {
        ctx.pi = intr.offset_ray_origin(intr.wo);
    } else if (pbrt::is_transmissive(flags) && !pbrt::is_reflective(flags)) {
        ctx.pi = intr.offset_ray_origin(-intr.wo);
    }

    // Choose a light source for the direct lighting calculation
    FloatType u = sampler->get_1d();
    auto sampled_light = base->light_sampler->sample(ctx, u);

    Point2f uLight = sampler->get_2d();
    if (!sampled_light) {
        return SampledSpectrum(0);
    }

    // Sample a point on the light source for direct lighting
    auto light = sampled_light->light;
    auto ls = light->sample_li(ctx, uLight, lambda);
    if (!ls || !ls->l.is_positive() || ls->pdf == 0) {
        return SampledSpectrum(0);
    }

    // Evaluate BSDF for light sample and check light visibility
    Vector3f wo = intr.wo;
    Vector3f wi = ls->wi;
    SampledSpectrum f = bsdf->f(wo, wi) * wi.abs_dot(intr.shading.n.to_vector3());

    if (!f.is_positive() || !base->unoccluded(intr, ls->p_light)) {
        return SampledSpectrum(0);
    }

    // Return light's contribution to reflected radiance
    FloatType pdf_light = sampled_light->p * ls->pdf;
    if (pbrt::is_delta_light(light->get_light_type())) {
        return ls->l * f / pdf_light;
    }

    // for non delta light
    FloatType pdf_bsdf = bsdf->pdf(wo, wi);
    FloatType weight_light = power_heuristic(1, pdf_light, 1, pdf_bsdf);

    return weight_light * ls->l * f / pdf_light;
}

void WavefrontPathIntegrator::render(Film *film, const bool preview) {
    printf("wavefront: path pool size: %u\n", PATH_POOL_SIZE);

    const auto image_resolution = this->path_state.image_resolution;

    const auto num_pixels = image_resolution.x * image_resolution.y;

    GLHelper gl_helper;
    if (preview) {
        gl_helper.init("initializing", image_resolution);
    }

    constexpr uint threads = 256;

    // generate new paths for the whole pool
    fill_new_path_queue<<<PATH_POOL_SIZE, threads>>>(&queues);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    queues.new_paths->counter = PATH_POOL_SIZE;
    queues.rays->counter = 0;

    generate_new_path<<<divide_and_ceil(queues.new_paths->counter, threads), threads>>>(
        &path_state, &queues, base);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    while (queues.rays->counter > 0) {
        ray_cast<<<divide_and_ceil(queues.rays->counter, threads), threads>>>(&path_state, &queues,
                                                                              base);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // clear all queues before control stage
        for (auto _queue : queues.get_all_queues()) {
            _queue->counter = 0;
        }
        queues.frame_buffer_counter = 0;

        control_logic<<<divide_and_ceil(PATH_POOL_SIZE, threads), threads>>>(&path_state, &queues,
                                                                             max_depth, base);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        if (queues.frame_buffer_counter > 0) {
            // sort to make film writing deterministic
            std::sort(queues.frame_buffer_queue + 0,
                      queues.frame_buffer_queue + queues.frame_buffer_counter, std::less{});

            write_frame_buffer<<<divide_and_ceil(queues.frame_buffer_counter, threads), threads>>>(
                film, &queues);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            if (preview) {
                film->copy_to_frame_buffer(gl_helper.gpu_frame_buffer);

                const auto current_sample_idx =
                    std::min<uint>(path_state.global_path_counter / num_pixels, samples_per_pixel);

                gl_helper.draw_frame(
                    GLHelper::assemble_title(FloatType(current_sample_idx) / samples_per_pixel));
            }
        }

        if (queues.new_paths->counter > 0) {
            generate_new_path<<<divide_and_ceil(queues.new_paths->counter, threads), threads>>>(
                &path_state, &queues, base);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }

        for (const auto material_type : Material::get_all_material_type()) {
            evaluate_material(material_type);
        }
    }
}

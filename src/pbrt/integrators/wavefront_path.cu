#include "pbrt/accelerator/hlbvh.h"
#include "pbrt/base/film.h"
#include "pbrt/base/integrator_base.h"
#include "pbrt/base/light.h"
#include "pbrt/base/material.h"
#include "pbrt/base/sampler.h"
#include "pbrt/gui/gl_object.h"
#include "pbrt/integrators/wavefront_path.h"
#include "pbrt/light_samplers/power_light_sampler.h"
#include "pbrt/samplers/independent.h"
#include "pbrt/samplers/stratified.h"
#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"
#include "pbrt/spectrum_util/sampled_wavelengths.h"
#include "pbrt/util/basic_math.h"

const uint PATH_POOL_SIZE = 2 * 1024 * 1024;

struct FrameBuffer {
    uint pixel_idx;
    uint sample_idx;
    SampledSpectrum radiance;
    SampledWavelengths lambda;
    FloatType weight;
};

struct FBComparator {
    bool operator()(FrameBuffer const &left, FrameBuffer const &right) const {
        if (left.pixel_idx < right.pixel_idx) {
            return true;
        }

        if (left.pixel_idx > right.pixel_idx) {
            return false;
        }

        return left.sample_idx < right.sample_idx;
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

static __global__ void gpu_init_path_state(PathState *path_state) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= PATH_POOL_SIZE) {
        return;
    }

    path_state->init_new_path(worker_idx);
}

__global__ void control_logic(const WavefrontPathIntegrator *integrator, PathState *path_state,
                              Queues *queues) {
    const uint path_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (path_idx >= PATH_POOL_SIZE || path_state->finished[path_idx]) {
        return;
    }

    // otherwise beta is larger than 0.0
    auto &isect = path_state->shape_intersections[path_idx].interaction;
    const auto ray = path_state->camera_rays[path_idx].ray;
    auto &lambda = path_state->lambdas[path_idx];

    const auto path_length = path_state->path_length[path_idx];
    const auto specular_bounce = path_state->mis_parameters[path_idx].specular_bounce;
    auto &beta = path_state->beta[path_idx];
    auto &L = path_state->L[path_idx];

    const auto intersected = path_state->intersected[path_idx];

    const auto prev_interaction_light_sample_ctx =
        path_state->mis_parameters[path_idx].prev_interaction_light_sample_ctx;
    const auto pdf_bsdf = path_state->mis_parameters[path_idx].pdf_bsdf;

    bool should_terminate_path =
        !intersected || path_length > integrator->max_depth || !beta.is_positive();

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
            for (uint idx = 0; idx < integrator->base->infinite_light_num; ++idx) {
                auto light = integrator->base->infinite_lights[idx];
                auto Le = light->le(ray, lambda);

                if (path_length == 0 || specular_bounce) {
                    L += beta * Le;
                } else {
                    // Compute MIS weight for infinite light
                    FloatType pdf_light =
                        integrator->base->light_sampler->pmf(prev_interaction_light_sample_ctx,
                                                             light) *
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

        queues->new_path_queue[atomicAdd(&queues->new_path_counter, 1)] = path_idx;
        return;
    }

    SampledSpectrum Le = isect.le(-ray.d, lambda);
    if (Le.is_positive()) {
        if (path_length == 0 || specular_bounce)
            path_state->L[path_idx] += beta * Le;
        else {
            // Compute MIS weight for area light
            auto area_light = isect.area_light;

            FloatType pdf_light = integrator->base->light_sampler->pmf(
                                      prev_interaction_light_sample_ctx, area_light) *
                                  area_light->pdf_li(prev_interaction_light_sample_ctx, ray.d);
            FloatType weight_light = power_heuristic(1, pdf_bsdf, 1, pdf_light);

            path_state->L[path_idx] += beta * weight_light * Le;
        }
    }

    // for active paths: advance one segment

    path_state->path_length[path_idx] += 1;

    switch (isect.material->get_material_type()) {

    case Material::Type::conductor: {
        const uint queue_idx = atomicAdd(&queues->conductor_material_counter, 1);
        queues->conductor_material_queue[queue_idx] = path_idx;
        break;
    }

    case Material::Type::coated_conductor: {
        const uint queue_idx = atomicAdd(&queues->coated_conductor_material_counter, 1);
        queues->coated_conductor_material_queue[queue_idx] = path_idx;
        break;
    }

    case Material::Type::coated_diffuse: {
        const uint queue_idx = atomicAdd(&queues->coated_diffuse_material_counter, 1);
        queues->coated_diffuse_material_queue[queue_idx] = path_idx;
        break;
    }

    case Material::Type::dielectric: {
        const uint queue_idx = atomicAdd(&queues->dielectric_material_counter, 1);
        queues->dielectric_material_queue[queue_idx] = path_idx;
        break;
    }

    case Material::Type::diffuse: {
        const uint queue_idx = atomicAdd(&queues->diffuse_material_counter, 1);
        queues->diffuse_material_queue[queue_idx] = path_idx;
        break;
    }

    case Material::Type::mix: {
        printf("\nyou should not see MixMaterial here\n\n");
        REPORT_FATAL_ERROR();
    }

    default: {
        REPORT_FATAL_ERROR();
    }
    }
}

__global__ void write_frame_buffer(Film *film, Queues *queues) {
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

__global__ void fill_new_path_queue(Queues *queues) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= PATH_POOL_SIZE) {
        return;
    }
    queues->new_path_queue[worker_idx] = worker_idx;
}

__global__ void generate_new_path(const IntegratorBase *base, PathState *path_state,
                                  Queues *queues) {
    const uint queue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_idx >= queues->new_path_counter) {
        return;
    }

    const uint path_idx = queues->new_path_queue[queue_idx];

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

    uint ray_queue_idx = atomicAdd(&queues->ray_counter, 1);
    queues->ray_queue[ray_queue_idx] = path_idx;
}

template <Material::Type material_type>
__global__ void gpu_evaluate_material(const WavefrontPathIntegrator *integrator,
                                      PathState *path_state, Queues *queues) {
    uint material_counter = 0;
    uint *material_queue = nullptr;

    switch (material_type) {
    case Material::Type::coated_conductor: {
        material_counter = queues->coated_conductor_material_counter;
        material_queue = queues->coated_conductor_material_queue;
        break;
    }

    case Material::Type::coated_diffuse: {
        material_counter = queues->coated_diffuse_material_counter;
        material_queue = queues->coated_diffuse_material_queue;
        break;
    }

    case Material::Type::conductor: {
        material_counter = queues->conductor_material_counter;
        material_queue = queues->conductor_material_queue;
        break;
    }

    case Material::Type::dielectric: {
        material_counter = queues->dielectric_material_counter;
        material_queue = queues->dielectric_material_queue;
        break;
    }

    case Material::Type::diffuse: {
        material_counter = queues->diffuse_material_counter;
        material_queue = queues->diffuse_material_queue;
        break;
    }

    default: {
        REPORT_FATAL_ERROR();
    }
    }

    const uint queue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (queue_idx >= material_counter) {
        return;
    }

    const uint path_idx = material_queue[queue_idx];

    auto &lambda = path_state->lambdas[path_idx];

    auto sampler = &path_state->samplers[path_idx];

    auto &isect = path_state->shape_intersections[path_idx].interaction;

    path_state->bsdf[path_idx] =
        isect.get_bsdf(lambda, integrator->base->camera, sampler->get_samples_per_pixel());

    integrator->sample_bsdf(path_idx, path_state);

    uint ray_queue_idx = atomicAdd(&queues->ray_counter, 1);
    queues->ray_queue[ray_queue_idx] = path_idx;
}

__global__ void ray_cast(const WavefrontPathIntegrator *integrator, PathState *path_state,
                         Queues *queues) {
    const uint ray_queue_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_queue_idx >= queues->ray_counter) {
        return;
    }

    const uint path_idx = queues->ray_queue[ray_queue_idx];

    const auto camera_ray = path_state->camera_rays[path_idx];

    auto intersection = integrator->base->intersect(camera_ray.ray, Infinity);

    path_state->intersected[path_idx] = intersection.has_value();

    if (intersection.has_value()) {
        path_state->shape_intersections[path_idx] = intersection.value();
    }
}

PBRT_GPU void WavefrontPathIntegrator::sample_bsdf(uint path_idx, PathState *path_state) const {
    auto &isect = path_state->shape_intersections[path_idx].interaction;
    auto &lambda = path_state->lambdas[path_idx];

    auto &ray = path_state->camera_rays[path_idx].ray;
    auto sampler = &path_state->samplers[path_idx];

    if (regularize && path_state->mis_parameters[path_idx].any_non_specular_bounces) {
        path_state->bsdf[path_idx].regularize();
    }

    if (_is_non_specular(path_state->bsdf[path_idx].flags())) {
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

template <Material::Type material_type>
void WavefrontPathIntegrator::evaluate_material() {
    uint material_counter = 0;
    switch (material_type) {
    case Material::Type::coated_conductor: {
        material_counter = queues.coated_conductor_material_counter;
        break;
    }

    case Material::Type::coated_diffuse: {
        material_counter = queues.coated_diffuse_material_counter;
        break;
    }

    case Material::Type::conductor: {
        material_counter = queues.conductor_material_counter;
        break;
    }

    case Material::Type::dielectric: {
        material_counter = queues.dielectric_material_counter;
        break;
    }

    case Material::Type::diffuse: {
        material_counter = queues.diffuse_material_counter;
        break;
    }

    default: {
        REPORT_FATAL_ERROR();
    }
    }

    if (material_counter <= 0) {
        return;
    }

    const uint threads = 256;
    const auto blocks = divide_and_ceil(material_counter, threads);

    gpu_evaluate_material<material_type><<<blocks, threads>>>(this, &path_state, &queues);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

PBRT_CPU_GPU
void PathState::init_new_path(uint path_idx) {
    intersected[path_idx] = false;
    finished[path_idx] = false;

    L[path_idx] = SampledSpectrum(0.0);
    beta[path_idx] = SampledSpectrum(1.0);
    path_length[path_idx] = 0;

    mis_parameters[path_idx].init();
}

void PathState::create(uint samples_per_pixel, const Point2i &_resolution,
                       const std::string &sampler_type, std::vector<void *> &gpu_dynamic_pointers) {
    image_resolution = _resolution;
    global_path_counter = 0;
    total_path_num = samples_per_pixel * image_resolution.x * image_resolution.y;

    CHECK_CUDA_ERROR(cudaMallocManaged(&camera_samples, sizeof(CameraSample) * PATH_POOL_SIZE));
    CHECK_CUDA_ERROR(cudaMallocManaged(&camera_rays, sizeof(CameraRay) * PATH_POOL_SIZE));
    CHECK_CUDA_ERROR(cudaMallocManaged(&lambdas, sizeof(SampledWavelengths) * PATH_POOL_SIZE));

    CHECK_CUDA_ERROR(cudaMallocManaged(&L, sizeof(SampledSpectrum) * PATH_POOL_SIZE));
    CHECK_CUDA_ERROR(cudaMallocManaged(&beta, sizeof(SampledSpectrum) * PATH_POOL_SIZE));
    CHECK_CUDA_ERROR(
        cudaMallocManaged(&shape_intersections, sizeof(ShapeIntersection) * PATH_POOL_SIZE));

    CHECK_CUDA_ERROR(cudaMallocManaged(&path_length, sizeof(uint) * PATH_POOL_SIZE));
    CHECK_CUDA_ERROR(cudaMallocManaged(&intersected, sizeof(bool) * PATH_POOL_SIZE));
    CHECK_CUDA_ERROR(cudaMallocManaged(&finished, sizeof(bool) * PATH_POOL_SIZE));
    CHECK_CUDA_ERROR(cudaMallocManaged(&pixel_indices, sizeof(uint) * PATH_POOL_SIZE));
    CHECK_CUDA_ERROR(cudaMallocManaged(&sample_indices, sizeof(uint) * PATH_POOL_SIZE));

    CHECK_CUDA_ERROR(cudaMallocManaged(&bsdf, sizeof(BSDF) * PATH_POOL_SIZE));

    CHECK_CUDA_ERROR(cudaMallocManaged(&mis_parameters, sizeof(MISParameter) * PATH_POOL_SIZE));

    CHECK_CUDA_ERROR(cudaMallocManaged(&samplers, sizeof(Sampler) * PATH_POOL_SIZE));

    for (auto ptr :
         std::vector<void *>({camera_samples, camera_rays, lambdas, L, beta, shape_intersections,
                              path_length, intersected, finished, pixel_indices, sample_indices,
                              bsdf, mis_parameters, samplers})) {
        gpu_dynamic_pointers.push_back(ptr);
    }

    const uint threads = 1024;
    uint blocks = divide_and_ceil<uint>(PATH_POOL_SIZE, threads);

    if (sampler_type == "stratified") {
        const uint samples_per_dimension = std::sqrt(samples_per_pixel);
        if (samples_per_dimension * samples_per_dimension != samples_per_pixel) {
            REPORT_FATAL_ERROR();
        }

        StratifiedSampler *stratified_samplers;
        CHECK_CUDA_ERROR(
            cudaMallocManaged(&stratified_samplers, sizeof(StratifiedSampler) * PATH_POOL_SIZE));
        gpu_dynamic_pointers.push_back(stratified_samplers);

        gpu_init_stratified_samplers<<<blocks, threads>>>(samplers, stratified_samplers,
                                                          samples_per_dimension, PATH_POOL_SIZE);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    } else if (sampler_type == "independent") {
        IndependentSampler *independent_samplers;
        CHECK_CUDA_ERROR(
            cudaMallocManaged(&independent_samplers, sizeof(IndependentSampler) * PATH_POOL_SIZE));
        gpu_dynamic_pointers.push_back(independent_samplers);

        gpu_init_independent_samplers<<<blocks, threads>>>(samplers, independent_samplers,
                                                           PATH_POOL_SIZE);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    } else {
        REPORT_FATAL_ERROR();
    }

    gpu_init_path_state<<<PATH_POOL_SIZE, threads>>>(this);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

void Queues::init(std::vector<void *> &gpu_dynamic_pointers) {
    CHECK_CUDA_ERROR(cudaMallocManaged(&new_path_queue, sizeof(uint) * PATH_POOL_SIZE));
    CHECK_CUDA_ERROR(cudaMallocManaged(&ray_queue, sizeof(uint) * PATH_POOL_SIZE));
    CHECK_CUDA_ERROR(cudaMallocManaged(&frame_buffer_queue, sizeof(FrameBuffer) * PATH_POOL_SIZE));

    CHECK_CUDA_ERROR(
        cudaMallocManaged(&coated_conductor_material_queue, sizeof(uint) * PATH_POOL_SIZE));
    CHECK_CUDA_ERROR(
        cudaMallocManaged(&coated_diffuse_material_queue, sizeof(uint) * PATH_POOL_SIZE));
    CHECK_CUDA_ERROR(cudaMallocManaged(&conductor_material_queue, sizeof(uint) * PATH_POOL_SIZE));
    CHECK_CUDA_ERROR(cudaMallocManaged(&dielectric_material_queue, sizeof(uint) * PATH_POOL_SIZE));
    CHECK_CUDA_ERROR(cudaMallocManaged(&diffuse_material_queue, sizeof(uint) * PATH_POOL_SIZE));

    for (auto ptr : std::vector<void *>({new_path_queue, ray_queue, frame_buffer_queue,
                                         coated_conductor_material_queue,
                                         coated_diffuse_material_queue, conductor_material_queue,
                                         dielectric_material_queue, diffuse_material_queue})) {
        gpu_dynamic_pointers.push_back(ptr);
    }
}

WavefrontPathIntegrator *
WavefrontPathIntegrator::create(const ParameterDictionary &parameters, const IntegratorBase *base,
                                const std::string &sampler_type, uint samples_per_pixel,
                                std::vector<void *> &gpu_dynamic_pointers) {
    WavefrontPathIntegrator *integrator;
    CHECK_CUDA_ERROR(cudaMallocManaged(&integrator, sizeof(WavefrontPathIntegrator)));
    gpu_dynamic_pointers.push_back(integrator);

    if (sampler_type == "stratified") {
        auto old_spp = samples_per_pixel;
        samples_per_pixel = sqr(int(std::sqrt(samples_per_pixel)));
        if (old_spp != samples_per_pixel) {
            printf("samples per pixel adjusted: %d -> %d\n", old_spp, samples_per_pixel);
        }
    }

    integrator->samples_per_pixel = samples_per_pixel;

    integrator->base = base;
    integrator->path_state.create(samples_per_pixel, base->camera->get_camerabase()->resolution,
                                  sampler_type, gpu_dynamic_pointers);
    integrator->queues.init(gpu_dynamic_pointers);

    integrator->max_depth = parameters.get_integer("maxdepth", 5);

    integrator->regularize = parameters.get_bool("regularize", false);

    return integrator;
}

PBRT_GPU
SampledSpectrum WavefrontPathIntegrator::sample_ld(const SurfaceInteraction &intr, const BSDF *bsdf,
                                                   SampledWavelengths &lambda,
                                                   Sampler *sampler) const {
    // Initialize _LightSampleContext_ for light sampling
    LightSampleContext ctx(intr);
    // Try to nudge the light sampling position to correct side of the surface
    BxDFFlags flags = bsdf->flags();
    if (_is_reflective(flags) && !_is_transmissive(flags)) {
        ctx.pi = intr.offset_ray_origin(intr.wo);
    } else if (_is_transmissive(flags) && !_is_reflective(flags)) {
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

    std::vector<void *> gpu_dynamic_pointers;
    uint8_t *gpu_frame_buffer = nullptr;
    GLObject gl_object;
    if (preview) {
        gl_object.init("initializing", image_resolution);

        CHECK_CUDA_ERROR(cudaMallocManaged(
            &gpu_frame_buffer, sizeof(uint8_t) * 3 * image_resolution.x * image_resolution.y));
        gpu_dynamic_pointers.push_back(gpu_frame_buffer);
    }

    constexpr uint threads = 256;

    // generate new paths for the whole pool
    fill_new_path_queue<<<PATH_POOL_SIZE, threads>>>(&queues);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    queues.new_path_counter = PATH_POOL_SIZE;

    queues.ray_counter = 0;
    generate_new_path<<<divide_and_ceil(queues.new_path_counter, threads), threads>>>(
        base, &path_state, &queues);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    while (queues.ray_counter > 0) {
        ray_cast<<<divide_and_ceil(queues.ray_counter, threads), threads>>>(this, &path_state,
                                                                            &queues);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // clear all queues before control stage
        queues.new_path_counter = 0;
        queues.ray_counter = 0;
        queues.frame_buffer_counter = 0;

        queues.coated_conductor_material_counter = 0;
        queues.coated_diffuse_material_counter = 0;
        queues.conductor_material_counter = 0;
        queues.dielectric_material_counter = 0;
        queues.diffuse_material_counter = 0;

        control_logic<<<divide_and_ceil(PATH_POOL_SIZE, threads), threads>>>(this, &path_state,
                                                                             &queues);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        if (queues.frame_buffer_counter > 0) {
            // sort to make film writing deterministic
            std::sort(queues.frame_buffer_queue + 0,
                      queues.frame_buffer_queue + queues.frame_buffer_counter, FBComparator());

            write_frame_buffer<<<divide_and_ceil(queues.frame_buffer_counter, threads), threads>>>(
                film, &queues);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            if (preview) {
                film->copy_to_frame_buffer(gpu_frame_buffer);

                auto current_sample_idx =
                    std::min<uint>(path_state.global_path_counter / num_pixels, samples_per_pixel);

                gl_object.draw_frame(
                    gpu_frame_buffer,
                    GLObject::assemble_title(FloatType(current_sample_idx) / samples_per_pixel),
                    image_resolution);
            }
        }

        if (queues.new_path_counter > 0) {
            generate_new_path<<<divide_and_ceil(queues.new_path_counter, threads), threads>>>(
                base, &path_state, &queues);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }

        evaluate_material<Material::Type::coated_conductor>();

        evaluate_material<Material::Type::coated_diffuse>();

        evaluate_material<Material::Type::conductor>();

        evaluate_material<Material::Type::dielectric>();

        evaluate_material<Material::Type::diffuse>();
    }

    for (auto ptr : gpu_dynamic_pointers) {
        CHECK_CUDA_ERROR(cudaFree(ptr));
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

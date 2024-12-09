#include "pbrt/base/film.h"
#include "pbrt/base/megakernel_integrator.h"
#include "pbrt/base/ray.h"
#include "pbrt/base/sampler.h"
#include "pbrt/integrators/ambient_occlusion.h"
#include "pbrt/integrators/path.h"
#include "pbrt/integrators/random_walk.h"
#include "pbrt/integrators/simple_path.h"
#include "pbrt/integrators/surface_normal.h"
#include "pbrt/spectrum_util/sampled_wavelengths.h"

PBRT_GPU
static void evaluate_pixel_sample(Film *film, const Point2i p_pixel, const uint samples_per_pixel,
                                  Sampler *samplers, const Integrator *integrator,
                                  const IntegratorBase *integrator_base) {
    auto resolution = film->get_resolution();
    int width = resolution.x;
    const uint pixel_index = p_pixel.y * width + p_pixel.x;

    auto local_sampler = &samplers[pixel_index];

    auto camera = integrator_base->camera;
    auto filter = integrator_base->filter;

    for (uint i = 0; i < samples_per_pixel; ++i) {
        local_sampler->start_pixel_sample(pixel_index, i, 0);
        auto camera_sample = local_sampler->get_camera_sample(p_pixel, filter);
        auto lu = local_sampler->get_1d();
        auto lambda = SampledWavelengths::sample_visible(lu);

        auto ray = camera->generate_ray(camera_sample, local_sampler);

        auto radiance_l = ray.weight * integrator->li(ray.ray, lambda, local_sampler);

        if constexpr (DEBUG_MODE && radiance_l.has_nan()) {
            printf("%s(): pixel(%d, %d), samples %u: has NAN\n", __func__, p_pixel.x, p_pixel.y, i);
        }

        film->add_sample(p_pixel, radiance_l, lambda, camera_sample.filter_weight);
    }
}

__global__ static void megakernel_render(Film *film, const uint samples_per_pixel,
                                         Sampler *samplers, const Integrator *integrator,
                                         const IntegratorBase *integrator_base) {
    auto resolution = film->get_resolution();

    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= resolution.x || y >= resolution.y) {
        return;
    }

    evaluate_pixel_sample(film, Point2i(x, y), samples_per_pixel, samplers, integrator,
                          integrator_base);
}

const Integrator *Integrator::create(const ParameterDictionary &parameters,
                                     const std::string &integrator_name,
                                     const IntegratorBase *integrator_base,
                                     std::vector<void *> &gpu_dynamic_pointers) {

    Integrator *integrator;
    CHECK_CUDA_ERROR(cudaMallocManaged(&integrator, sizeof(Integrator)));
    gpu_dynamic_pointers.push_back(integrator);

    if (integrator_name == "ambientocclusion") {
        auto ambient_occlusion_integrator =
            AmbientOcclusionIntegrator::create(parameters, integrator_base, gpu_dynamic_pointers);

        integrator->init(ambient_occlusion_integrator);
        return integrator;
    }

    if (integrator_name == "path") {
        auto path_integrator =
            PathIntegrator::create(parameters, integrator_base, gpu_dynamic_pointers);

        integrator->init(path_integrator);
        return integrator;
    }

    if (integrator_name == "surfacenormal") {
        auto surface_normal_integrator =
            SurfaceNormalIntegrator::create(parameters, integrator_base, gpu_dynamic_pointers);

        integrator->init(surface_normal_integrator);
        return integrator;
    }

    if (integrator_name == "simplepath") {
        auto simple_path_integrator =
            SimplePathIntegrator::create(parameters, integrator_base, gpu_dynamic_pointers);

        integrator->init(simple_path_integrator);
        return integrator;
    }

    printf("\n%s(): unknown Integrator: %s\n\n", __func__, integrator_name.c_str());
    REPORT_FATAL_ERROR();
    return nullptr;
}

void Integrator::init(const AmbientOcclusionIntegrator *ambient_occlusion_integrator) {
    type = Type::ambient_occlusion;
    ptr = ambient_occlusion_integrator;
}

void Integrator::init(const PathIntegrator *path_integrator) {
    type = Type::path;
    ptr = path_integrator;
}

void Integrator::init(const RandomWalkIntegrator *random_walk_integrator) {
    type = Type::random_walk;
    ptr = random_walk_integrator;
}

void Integrator::init(const SurfaceNormalIntegrator *surface_normal_integrator) {
    type = Type::surface_normal;
    ptr = surface_normal_integrator;
}

void Integrator::init(const SimplePathIntegrator *simple_path_integrator) {
    type = Type::simple_path;
    ptr = simple_path_integrator;
}

PBRT_GPU
SampledSpectrum Integrator::li(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler) const {
    switch (type) {
    case Type::ambient_occlusion: {
        return static_cast<const AmbientOcclusionIntegrator *>(ptr)->li(ray, lambda, sampler);
    }

    case Type::path: {
        return static_cast<const PathIntegrator *>(ptr)->li(ray, lambda, sampler);
    }

    case Type::random_walk: {
        return static_cast<const RandomWalkIntegrator *>(ptr)->li(ray, lambda, sampler);
    }

    case Type::simple_path: {
        return static_cast<const SimplePathIntegrator *>(ptr)->li(ray, lambda, sampler);
    }

    case Type::surface_normal: {
        return static_cast<const SurfaceNormalIntegrator *>(ptr)->li(ray, lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

void Integrator::render(Film *film, const std::string &sampler_type, uint samples_per_pixel,
                        const IntegratorBase *integrator_base) const {
    auto film_resolution = integrator_base->camera->get_camerabase()->resolution;

    std::vector<void *> gpu_dynamic_pointers;
    auto samplers = Sampler::create(sampler_type, samples_per_pixel,
                                    film_resolution.x * film_resolution.y, gpu_dynamic_pointers);

    const uint thread_width = 8;
    const uint thread_height = 8;

    std::cout << " (samples per pixel: " << samples_per_pixel << ") "
              << "in " << thread_width << "x" << thread_height << " blocks.\n"
              << std::flush;

    dim3 blocks(divide_and_ceil(uint(film_resolution.x), thread_width),
                divide_and_ceil(uint(film_resolution.y), thread_height), 1);
    dim3 threads(thread_width, thread_height, 1);

    megakernel_render<<<blocks, threads>>>(film, samples_per_pixel, samplers, this,
                                           integrator_base);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    for (auto ptr : gpu_dynamic_pointers) {
        CHECK_CUDA_ERROR(cudaFree(ptr));
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

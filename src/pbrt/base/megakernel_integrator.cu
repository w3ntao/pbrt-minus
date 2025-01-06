#include "pbrt/base/film.h"
#include "pbrt/base/megakernel_integrator.h"
#include "pbrt/base/sampler.h"
#include "pbrt/gui/gl_object.h"
#include "pbrt/integrators/ambient_occlusion.h"
#include "pbrt/integrators/megakernel_path.h"
#include "pbrt/integrators/surface_normal.h"
#include "pbrt/spectrum_util/color_encoding.h"
#include <chrono>
#include <numeric>
#include <thread>

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

__global__ static void megakernel_render(Film *film, uint8_t *gpu_frame_buffer, bool *status_map,
                                         const uint samples_per_pixel, Sampler *samplers,
                                         const Integrator *integrator,
                                         const IntegratorBase *integrator_base) {
    const auto resolution = film->get_resolution();

    const auto x = threadIdx.x + blockIdx.x * blockDim.x;
    const auto y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= resolution.x || y >= resolution.y) {
        return;
    }

    evaluate_pixel_sample(film, Point2i(x, y), samples_per_pixel, samplers, integrator,
                          integrator_base);

    if (gpu_frame_buffer == nullptr || status_map == nullptr) {
        return;
    }

    const auto pixel_idx = y * resolution.x + x;

    const auto rgb = film->get_pixel_rgb(Point2i(x, y));
    if (rgb.has_nan()) {
        gpu_frame_buffer[pixel_idx * 3 + 0] = 0;
        gpu_frame_buffer[pixel_idx * 3 + 1] = 0;
        gpu_frame_buffer[pixel_idx * 3 + 2] = 0;

        return;
    }

    const SRGBColorEncoding srgb_encoding;

    gpu_frame_buffer[pixel_idx * 3 + 0] = srgb_encoding.from_linear(rgb.r);
    gpu_frame_buffer[pixel_idx * 3 + 1] = srgb_encoding.from_linear(rgb.g);
    gpu_frame_buffer[pixel_idx * 3 + 2] = srgb_encoding.from_linear(rgb.b);

    status_map[pixel_idx] = true;
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

    if (integrator_name == "megakernelpath") {
        auto path_integrator =
            MegakernelPathIntegrator::create(parameters, integrator_base, gpu_dynamic_pointers);
        integrator->init(path_integrator);

        return integrator;
    }

    if (integrator_name == "surfacenormal") {
        auto surface_normal_integrator =
            SurfaceNormalIntegrator::create(parameters, integrator_base, gpu_dynamic_pointers);
        integrator->init(surface_normal_integrator);

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

void Integrator::init(const MegakernelPathIntegrator *megakernel_path_integrator) {
    type = Type::megakernel_path;
    ptr = megakernel_path_integrator;
}

void Integrator::init(const SurfaceNormalIntegrator *surface_normal_integrator) {
    type = Type::surface_normal;
    ptr = surface_normal_integrator;
}

PBRT_GPU
SampledSpectrum Integrator::li(const Ray &ray, SampledWavelengths &lambda, Sampler *sampler) const {
    switch (type) {
    case Type::ambient_occlusion: {
        return static_cast<const AmbientOcclusionIntegrator *>(ptr)->li(ray, lambda, sampler);
    }

    case Type::megakernel_path: {
        return static_cast<const MegakernelPathIntegrator *>(ptr)->li(ray, lambda, sampler);
    }

    case Type::surface_normal: {
        return static_cast<const SurfaceNormalIntegrator *>(ptr)->li(ray, lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

void Integrator::render(Film *film, const std::string &sampler_type, const uint samples_per_pixel,
                        const IntegratorBase *integrator_base, const bool preview) const {
    const auto film_resolution = integrator_base->camera->get_camerabase()->resolution;
    const auto num_pixels = film_resolution.x * film_resolution.y;

    std::vector<void *> gpu_dynamic_pointers;
    auto samplers =
        Sampler::create(sampler_type, samples_per_pixel, num_pixels, gpu_dynamic_pointers);

    const uint thread_width = 8;
    const uint thread_height = 8;

    std::cout << " (samples per pixel: " << samples_per_pixel << ") "
              << "in " << thread_width << "x" << thread_height << " blocks.\n"
              << std::flush;

    GLObject gl_object;
    uint8_t *gpu_frame_buffer = nullptr;
    bool *status_map = nullptr;
    if (preview) {
        gl_object.init("initializing", film_resolution);
        CHECK_CUDA_ERROR(cudaMallocManaged(&status_map, sizeof(bool) * num_pixels));
        CHECK_CUDA_ERROR(cudaMallocManaged(&gpu_frame_buffer, sizeof(uint8_t) * 3 * num_pixels));

        gpu_dynamic_pointers.push_back(status_map);
        gpu_dynamic_pointers.push_back(gpu_frame_buffer);

        for (uint idx = 0; idx < num_pixels; ++idx) {
            status_map[idx] = false;
            gpu_frame_buffer[idx * 3 + 0] = 0;
            gpu_frame_buffer[idx * 3 + 1] = 0;
            gpu_frame_buffer[idx * 3 + 2] = 0;
        }
    }

    dim3 blocks(divide_and_ceil(uint(film_resolution.x), thread_width),
                divide_and_ceil(uint(film_resolution.y), thread_height), 1);
    dim3 threads(thread_width, thread_height, 1);
    megakernel_render<<<blocks, threads>>>(film, gpu_frame_buffer, status_map, samples_per_pixel,
                                           samplers, this, integrator_base);

    if (preview) {
        while (true) {
            const int finished_pixels = std::accumulate(status_map, status_map + num_pixels, 0);

            gl_object.draw_frame(gpu_frame_buffer,
                                 GLObject::assemble_title(double(finished_pixels) / num_pixels),
                                 film_resolution);

            if (finished_pixels >= num_pixels) {
                break;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    for (auto ptr : gpu_dynamic_pointers) {
        CHECK_CUDA_ERROR(cudaFree(ptr));
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
}

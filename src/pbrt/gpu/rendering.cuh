#pragma once

#include <iostream>
#include <string>
#include <iomanip>

#include <curand_kernel.h>

#include "ext/lodepng/lodepng.h"

#include "pbrt/cameras/perspective.h"
#include "pbrt/shapes/triangle.h"
#include "pbrt/integrators/surface_normal.h"
#include "pbrt/integrators/ambient_occlusion.h"
#include "pbrt/samplers/independent.h"
#include "pbrt/spectra/color_encoding.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

inline void check_cuda(cudaError_t result, char const *const func, const char *const file,
                       int const line) {
    if (!result) {
        return;
    }

    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":"
              << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(-1);
}

namespace GPU {
class Renderer {
  public:
    const Integrator *integrator = nullptr;
    const Camera *camera = nullptr;
    Aggregate *aggregate = nullptr;

    PBRT_GPU ~Renderer() {
        delete integrator;
        delete aggregate;
        delete camera;
    }
};

__global__ void gpu_init_renderer(Renderer *renderer) {
    *renderer = Renderer();
    renderer->aggregate = new Aggregate();
}

__global__ void gpu_init_integrator(Renderer *renderer) {
    // Integrator *gpu_integrator = new SurfaceNormalIntegrator();
    Integrator *gpu_integrator = new AmbientOcclusionIntegrator();
    renderer->integrator = gpu_integrator;
}

__global__ void gpu_init_camera(Renderer *renderer, Point2i resolution,
                                const CameraTransform camera_transform, const double fov) {
    renderer->camera = new PerspectiveCamera(resolution, camera_transform, fov);
}

__global__ void gpu_aggregate_preprocess(Renderer *renderer) {
    renderer->aggregate->preprocess();
}

__global__ void gpu_add_triangle_mesh(Renderer *renderer, const Transform render_from_object,
                                      bool reverse_orientation, const Point3f *points,
                                      int num_points, const int *indicies, int num_indicies,
                                      const Point2f *uv, int num_uv) {
    const TriangleMesh *mesh = new TriangleMesh(render_from_object, reverse_orientation, indicies,
                                                num_indicies, points, num_points);
    renderer->aggregate->add_triangles(mesh);
}

__global__ void gpu_free_renderer(Renderer *renderer) {
    renderer->~Renderer();
    // renderer was never new in divice code
    // so you have to destruct it manually
}

__global__ void gpu_parallel_render(RGB *frame_buffer, int num_samples, const Renderer *renderer) {
    const Camera *camera = renderer->camera;

    int width = camera->resolution.x;
    int height = camera->resolution.y;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    int pixel_index = y * width + x;

    const Integrator *integrator = renderer->integrator;
    const Aggregate *aggregate = renderer->aggregate;
    auto sampler = IndependentSampler(pixel_index);

    auto accumulated_l = RGB(0);
    for (int i = 0; i < num_samples; ++i) {
        auto sampled_p_film = Point2f(x, y) + Point2f(0.5, 0.5) + sampler.get_2d();
        const Ray ray = camera->generate_ray(sampled_p_film);
        accumulated_l += integrator->li(ray, aggregate, &sampler);
    }

    frame_buffer[pixel_index] = accumulated_l / double(num_samples);
}

void writer_to_file(const std::string &filename, const RGB *frame_buffer, int width, int height) {
    SRGBColorEncoding srgb_encoding;
    std::vector<unsigned char> pixels(width * height * 4);

    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            const auto &color = frame_buffer[y * width + x];

            pixels[4 * (width * y + x) + 0] = srgb_encoding.from_linear(color.r);
            pixels[4 * (width * y + x) + 1] = srgb_encoding.from_linear(color.g);
            pixels[4 * (width * y + x) + 2] = srgb_encoding.from_linear(color.b);
            pixels[4 * (width * y + x) + 3] = 255;
        }
    }

    // Encode the image
    // if there's an error, display it
    if (unsigned error = lodepng::encode(filename, pixels, width, height); error) {
        std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error)
                  << std::endl;
        throw std::runtime_error("lodepng::encode() fail");
    }
}
} // namespace GPU

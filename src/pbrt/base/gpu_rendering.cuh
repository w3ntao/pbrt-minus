#pragma once

#include <iostream>
#include <string>
#include <iomanip>

#include <curand_kernel.h>

#include "pbrt/base/integrator.h"
#include "pbrt/cameras/perspective.h"
#include "pbrt/shapes/triangle.h"
#include "pbrt/integrators/path.h"
#include "pbrt/integrators/surface_normal.h"
#include "ext/lodepng/lodepng.h"

enum IntegratorType { PATH, SURFACE_NORMAL };

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file,
                int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":"
                  << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(-1);
    }
}

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

__global__ void init_gpu_renderer(Renderer *renderer) {
    *renderer = Renderer();
    renderer->aggregate = new Aggregate();
}

__global__ void init_gpu_integrator(Renderer *renderer) {
    Integrator *gpu_integrator = new SurfaceNormalIntegrator();
    renderer->integrator = gpu_integrator;
}

__global__ void init_gpu_camera(Renderer *renderer, Point2i resolution,
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

__global__ void free_renderer(Renderer *renderer) {
    renderer->~Renderer();
    // renderer was never new in divice code
    // so you have to destruct it manually
}

__global__ void gpu_render(Color *frame_buffer, int num_samples, const Renderer *renderer) {
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

    curandState local_rand_state;
    curand_init(1984, pixel_index, 0, &local_rand_state);

    // TODO: no random value here
    auto sampled_p_film = Point2f(x, y) + Point2f(0.5, 0.5);
    const Ray ray = camera->generate_ray(sampled_p_film);

    frame_buffer[pixel_index] = integrator->get_radiance(ray, aggregate, &local_rand_state);
}

void writer_to_file(const std::string &filename, const Color *frame_buffer, int width, int height) {
    std::vector<unsigned char> pixels(width * height * 4);

    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            const auto &color = frame_buffer[y * width + x];

            pixels[4 * (width * y + x) + 0] = (unsigned char)(color.r * 256);
            pixels[4 * (width * y + x) + 1] = (unsigned char)(color.g * 256);
            pixels[4 * (width * y + x) + 2] = (unsigned char)(color.b * 256);
            pixels[4 * (width * y + x) + 3] = 255;
        }
    }

    // Encode the image
    unsigned error = lodepng::encode(filename, pixels, width, height);
    // if there's an error, display it
    if (error) {
        std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error)
                  << std::endl;
        throw std::runtime_error("lodepng::encode() fail");
    }
}

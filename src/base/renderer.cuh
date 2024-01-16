#pragma once

#include "renderer.cuh"

#include <iostream>
#include <string>
#include <iomanip>

#include <curand_kernel.h>

#include "util/image.h"
#include "base/world.h"
#include "base/material.h"
#include "base/integrator.h"
#include "cameras/perspective.h"
#include "shapes/sphere.h"
#include "shapes/triangle.h"
#include "integrators/path.h"
#include "integrators/surface_normal.h"

const int shape_num = 2;

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
        int width = -1;
        int height = -1;
        const Integrator *integrator = nullptr;
        const World *world = nullptr;
        const Camera *camera = nullptr;

        PBRT_GPU Renderer(int _width, int _height) : width(_width), height(_height) {}

        PBRT_GPU ~Renderer() {
            if (integrator) {
                delete integrator;
            }

            if (world) {
                delete world;
            }

            if (camera) {
                delete camera;
            }
        }
};

__global__ void init_gpu_renderer(Renderer *gpu_renderer, IntegratorType type, int width,
                                  int height) {

    (*gpu_renderer) = Renderer(width, height);

    const Transform matrix_translate = Transform::translate(Vector3f(0, 0, -140));

    int *vertex_indices = new int[6];
    vertex_indices[0] = 0;
    vertex_indices[1] = 1;
    vertex_indices[2] = 2;
    vertex_indices[3] = 2;
    vertex_indices[4] = 3;
    vertex_indices[5] = 0;

    Point3f *points = new Point3f[4];
    points[0] = Point3f(-400, -400, 0);
    points[1] = Point3f(400, -400, 0);
    points[2] = Point3f(400, 400, 0);
    points[3] = Point3f(-400, 400, 0);

    const TriangleMesh *mesh = new TriangleMesh(matrix_translate, vertex_indices, 6, points, 4);

    delete[] vertex_indices;
    delete[] points;

    World *gpu_world = new World(shape_num);
    gpu_world->add_triangles(mesh);
    gpu_renderer->world = gpu_world;

    Integrator *gpu_integrator = nullptr;
    switch (type) {
    case IntegratorType::SURFACE_NORMAL: {
        gpu_integrator = new SurfaceNormalIntegrator();
        break;
    }
    case IntegratorType::PATH: {
        gpu_integrator = new PathIntegrator();
        break;
    }
    default: {
        asm("trap;");
        break;
    }
    }

    gpu_renderer->integrator = gpu_integrator;

    Point3f look_from(13, 2, 3);
    Point3f look_at(0, 0, 0);
    double dist_to_focus = (look_from - look_at).length();
    double aperture = 0.1;

    gpu_renderer->camera =
        new PerspectiveCamera(look_from, look_at, Vector3f(0, 1, 0), 30.0,
                              double(width) / double(height), aperture, dist_to_focus);
}

__global__ void free_renderer(Renderer *renderer) {
    renderer->~Renderer();
    // renderer was never new in divice code
    // so you have to destruct it manually
}

__global__ void gpu_render(Color *frame_buffer, int num_samples, const Renderer *renderer) {
    int width = renderer->width;
    int height = renderer->height;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    int pixel_index = y * width + x;

    const Integrator *integrator = renderer->integrator;
    const Camera *camera = renderer->camera;
    const World *world = renderer->world;

    frame_buffer[pixel_index] = Color(double(x) / (width - 1), double(y) / (height - 1), 0.0);

    return;

    curandState *local_rand_state = new curandState();
    curand_init(1984, pixel_index, 0, local_rand_state);

    Color final_color(0, 0, 0);
    for (int s = 0; s < num_samples; s++) {
        double u = double(x + curand_uniform(local_rand_state)) / double(width);
        double v = double(y + curand_uniform(local_rand_state)) / double(height);
        final_color += integrator->get_radiance(camera->get_ray(u, v, local_rand_state), world,
                                                local_rand_state);
    }
    delete local_rand_state;

    final_color /= double(num_samples);

    final_color = Color(sqrt(final_color.r), sqrt(final_color.g), sqrt(final_color.b));

    frame_buffer[pixel_index] = final_color;
}

void writer_to_file(const std::string &file_name, int width, int height,
                    const Color *frame_buffer) {
    Image image(frame_buffer, width, height);
    image.flip();
    image.writePNG(file_name);
}

void render(int num_samples, const std::string &file_name) {
    int width = 1960;
    int height = 1080;

    int thread_width = 8;
    int thread_height = 8;

    std::cerr << "Rendering a " << width << "x" << height
              << " image (samples per pixel: " << num_samples << ") ";
    std::cerr << "in " << thread_width << "x" << thread_height << " blocks.\n";

    Renderer *gpu_renderer;
    checkCudaErrors(cudaMallocManaged((void **)&gpu_renderer, sizeof(Renderer)));
    init_gpu_renderer<<<1, 1>>>(gpu_renderer, IntegratorType::SURFACE_NORMAL, width, height);

    // allocate FB
    Color *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, sizeof(Color) * width * height));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    clock_t start = clock();
    dim3 blocks(width / thread_width + 1, height / thread_height + 1, 1);
    dim3 threads(thread_width, thread_height, 1);

    gpu_render<<<blocks, threads>>>(frame_buffer, num_samples, gpu_renderer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    const double timer_seconds = (double)(clock() - start) / CLOCKS_PER_SEC;
    std::cerr << std::fixed << std::setprecision(1) << "took " << timer_seconds << " seconds.\n";

    writer_to_file(file_name, width, height, frame_buffer);

    free_renderer<<<1, 1>>>(gpu_renderer);
    checkCudaErrors(cudaFree(gpu_renderer));
    checkCudaErrors(cudaFree(frame_buffer));

    cudaDeviceReset();

    std::cout << "image saved to `" << file_name << "`\n";
}

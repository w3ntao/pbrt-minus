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
    const World *world = nullptr;
    const Camera *camera = nullptr;

    PBRT_GPU ~Renderer() {
        delete integrator;
        delete world;
        delete camera;
    }
};

__global__ void init_gpu_renderer(Renderer *renderer) {
    *renderer = Renderer();
}

__global__ void init_gpu_integrator(Renderer *renderer) {
    Integrator *gpu_integrator = new SurfaceNormalIntegrator();
    renderer->integrator = gpu_integrator;
}

__global__ void init_gpu_camera(Renderer *renderer, int width, int height) {
    Point3f look_from(200, 250, 70);
    Point3f look_at(0, 33, -50);
    Vector3f up(0, 0, 1);
    double dist_to_focus = (look_from - look_at).length();
    double aperture = 0.1;

    renderer->camera =
        new PerspectiveCamera(width, height, look_from, look_at, up, 30.0,
                              double(width) / double(height), aperture, dist_to_focus);
}

__global__ void init_gpu_world(Renderer *renderer, int num_primitive) {

    const Transform matrix_translate = Transform::translate(Vector3f(0, 0, -140));

    int *v_idx_0 = new int[6];
    v_idx_0[0] = 0;
    v_idx_0[1] = 1;
    v_idx_0[2] = 2;
    v_idx_0[3] = 2;
    v_idx_0[4] = 3;
    v_idx_0[5] = 0;

    Point3f *p0 = new Point3f[4];
    p0[0] = Point3f(-400, -400, 0);
    p0[1] = Point3f(400, -400, 0);
    p0[2] = Point3f(400, 400, 0);
    p0[3] = Point3f(-400, 400, 0);

    int *v_idx_1 = new int[6];
    v_idx_1[0] = 0;
    v_idx_1[1] = 1;
    v_idx_1[2] = 2;
    v_idx_1[3] = 2;
    v_idx_1[4] = 3;
    v_idx_1[5] = 0;

    Point3f *p1 = new Point3f[4];
    p1[0] = Point3f(-400, -400, 0);
    p1[1] = Point3f(400, -400, 0);
    p1[2] = Point3f(400, -400, 1000);
    p1[3] = Point3f(-400, -400, 1000);

    int *v_idx_2 = new int[6];
    v_idx_2[0] = 0;
    v_idx_2[1] = 1;
    v_idx_2[2] = 2;
    v_idx_2[3] = 2;
    v_idx_2[4] = 3;
    v_idx_2[5] = 0;

    Point3f *p2 = new Point3f[4];
    p2[0] = Point3f(-400, -400, 0);
    p2[1] = Point3f(-400, 400, 0);
    p2[2] = Point3f(-400, 400, 1000);
    p2[3] = Point3f(-400, -400, 1000);

    const TriangleMesh *mesh0 = new TriangleMesh(matrix_translate, v_idx_0, 6, p0, 4);
    const TriangleMesh *mesh1 = new TriangleMesh(matrix_translate, v_idx_1, 6, p1, 4);
    const TriangleMesh *mesh2 = new TriangleMesh(matrix_translate, v_idx_2, 6, p2, 4);

    delete[] v_idx_0;
    delete[] v_idx_1;
    delete[] v_idx_2;
    delete[] p0;
    delete[] p1;
    delete[] p2;

    World *world = new World(num_primitive);
    world->add_triangles(mesh0);
    world->add_triangles(mesh1);
    world->add_triangles(mesh2);
    renderer->world = world;
}

__global__ void free_renderer(Renderer *renderer) {
    renderer->~Renderer();
    // renderer was never new in divice code
    // so you have to destruct it manually
}

__global__ void gpu_render(Color *frame_buffer, int num_samples, const Renderer *renderer) {
    const Camera *camera = renderer->camera;

    int width = camera->width;
    int height = camera->height;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    int pixel_index = y * width + x;

    const Integrator *integrator = renderer->integrator;
    const World *world = renderer->world;

    curandState local_rand_state;
    curand_init(1984, pixel_index, 0, &local_rand_state);

    Color final_color(0, 0, 0);
    for (int s = 0; s < num_samples; s++) {
        double u = double(x + curand_uniform(&local_rand_state)) / double(width);
        double v = double(y + curand_uniform(&local_rand_state)) / double(height);
        final_color += integrator->get_radiance(camera->get_ray(u, v, &local_rand_state), world,
                                                &local_rand_state);
    }

    final_color /= double(num_samples);

    final_color = Color(sqrt(final_color.r), sqrt(final_color.g), sqrt(final_color.b));

    frame_buffer[pixel_index] = final_color;
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

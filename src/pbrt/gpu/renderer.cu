#include "pbrt/gpu/renderer.h"

__global__ static void parallel_render(Renderer *renderer, uint num_samples) {
    auto camera_base = renderer->camera->get_camerabase();

    uint width = camera_base->resolution.x;
    uint height = camera_base->resolution.y;

    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    renderer->evaluate_pixel_sample(Point2i(x, y), num_samples);
}

void Renderer::render(const std::string &output_filename, const Point2i &film_resolution,
                      uint samples_per_pixel) {
    uint thread_width = 8;
    uint thread_height = 8;

    std::cout << "\n";
    std::cout << "rendering a " << film_resolution.x << "x" << film_resolution.y
              << " image (samples per pixel: " << samples_per_pixel << ") ";
    std::cout << "in " << thread_width << "x" << thread_height << " blocks.\n" << std::flush;

    dim3 blocks(divide_and_ceil(uint(film_resolution.x), thread_width),
                divide_and_ceil(uint(film_resolution.y), thread_height), 1);
    dim3 threads(thread_width, thread_height, 1);

    parallel_render<<<blocks, threads>>>(this, samples_per_pixel);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    film->write_to_png(output_filename, film_resolution);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

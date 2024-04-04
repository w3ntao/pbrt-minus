#pragma once

#include <iostream>
#include <string>

#include <curand_kernel.h>
#include <cuda/atomic>

#include "ext/lodepng/lodepng.h"

#include "pbrt/base/shape.h"
#include "pbrt/base/spectrum.h"

#include "pbrt/spectra/constants.h"
#include "pbrt/spectra/color_encoding.h"
#include "pbrt/spectra/rgb_color_space.h"
#include "pbrt/spectra/sampled_wavelengths.h"
#include "pbrt/spectra/densely_sampled_spectrum.h"

#include "pbrt/accelerator/hlbvh.h"

#include "pbrt/shapes/triangle.h"

#include "pbrt/filters/box.h"

#include "pbrt/films/pixel_sensor.h"
#include "pbrt/films/rgb_film.h"

#include "pbrt/cameras/perspective.h"

#include "pbrt/integrators/surface_normal.h"
#include "pbrt/integrators/ambient_occlusion.h"

#include "pbrt/samplers/independent.h"

namespace GPU {

struct GlobalVariable {
    PBRT_CPU_GPU
    void init(const Spectrum *_cie_xyz[3], const Spectrum *cie_illum_d6500,
              const RGBtoSpectrumData::RGBtoSpectrumTableGPU *rgb_to_spectrum_table,
              RGBtoSpectrumData::Gamut gamut) {
        for (uint idx = 0; idx < 3; idx++) {
            cie_xyz[idx] = _cie_xyz[idx];
        }

        if (gamut == RGBtoSpectrumData::Gamut::srgb) {
            rgb_color_space->init(Point2f(0.64, 0.33), Point2f(0.3, 0.6), Point2f(0.15, 0.06),
                                  cie_illum_d6500, rgb_to_spectrum_table, cie_xyz);

            return;
        }

        printf("\nthis color space is not implemented\n\n");
        asm("trap;");
    }

    PBRT_CPU_GPU void get_cie_xyz(const Spectrum *out[3]) const {
        for (uint idx = 0; idx < 3; idx++) {
            out[idx] = cie_xyz[idx];
        }
    }

    RGBColorSpace *rgb_color_space;
    const Spectrum *cie_xyz[3];
};

class Renderer {
  public:
    Integrator *integrator;
    PerspectiveCamera *camera;
    BoxFilter *filter;
    RGBFilm *film;
    HLBVH *bvh;

    const GlobalVariable *global_variables;
    // TODO: move GlobalVariable* to Builder

    PixelSensor sensor;
    // TODO: change PixelSensor to PixelSensor*

    PBRT_GPU void evaluate_pixel_sample(const Point2i p_pixel, const int num_samples) {
        int width = camera->camera_base.resolution.x;
        int pixel_index = p_pixel.y * width + p_pixel.x;

        auto sampler = IndependentSampler(pixel_index);

        for (uint i = 0; i < num_samples; ++i) {
            auto camera_sample = sampler.get_camera_sample(p_pixel, filter);
            auto lu = sampler.get_1d();
            auto lambda = SampledWavelengths::sample_visible(lu);

            auto ray = camera->generate_ray(camera_sample);

            auto radiance_l = ray.weight * integrator->li(ray.ray, lambda, bvh, sampler);

            if (radiance_l.has_nan()) {
                printf("evaluate_pixel_sample(): pixel(%d, %d), samples %u: has an NAN component\n",
                       p_pixel.x, p_pixel.y, i);
            }

            film->add_sample(p_pixel, radiance_l, lambda, camera_sample.filter_weight);
        }
    }
};

template <typename S>
static __global__ void build_shapes(Shape *shapes, const S *concrete_shapes, uint num) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num) {
        return;
    }

    shapes[worker_idx].init(&concrete_shapes[worker_idx]);
}

template <typename T>
static __global__ void apply_transform(T *data, const Transform transform, uint length) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= length) {
        return;
    }

    data[idx] = transform(data[idx]);
}

__global__ void init_rgb_to_spectrum_table_coefficients(
    RGBtoSpectrumData::RGBtoSpectrumTableGPU *rgb_to_spectrum_data,
    const double *rgb_to_spectrum_table_coefficients) {
    /*
     * max thread size: 1024
     * total dimension: 3 * 64 * 64 * 64 * 3
     * 3: blocks.x
     * 64: blocks.y
     * 64: blocks.z
     * 64: threads.x
     * 3:  threads.y
     */

    constexpr int resolution = RGBtoSpectrumData::RES;

    uint max_component = blockIdx.x;
    uint z = blockIdx.y;
    uint y = blockIdx.z;

    uint x = threadIdx.x;
    uint c = threadIdx.y;

    uint idx = (((max_component * resolution + z) * resolution + y) * resolution + x) * 3 + c;

    rgb_to_spectrum_data->coefficients[max_component][z][y][x][c] =
        rgb_to_spectrum_table_coefficients[idx];
}

__global__ void
init_rgb_to_spectrum_table_scale(RGBtoSpectrumData::RGBtoSpectrumTableGPU *rgb_to_spectrum_data,
                                 const double *rgb_to_spectrum_table_scale) {
    uint idx = threadIdx.x;
    rgb_to_spectrum_data->z_nodes[idx] = rgb_to_spectrum_table_scale[idx];
}

static __global__ void init_pixels(Pixel *pixels, Point2i dimension) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= dimension.x * dimension.y) {
        return;
    }

    pixels[idx].init_zero();
}

__global__ void hlbvh_init_morton_primitives(HLBVH *bvh) {
    bvh->init_morton_primitives();
}

__global__ void hlbvh_init_treelets(HLBVH *bvh, Treelet *treelets) {
    bvh->init_treelets(treelets);
}

__global__ void hlbvh_compute_morton_code(HLBVH *bvh, const Bounds3f bounds_of_centroids) {
    bvh->compute_morton_code(bounds_of_centroids);
}

__global__ void hlbvh_build_treelets(HLBVH *bvh, Treelet *treelets) {
    bvh->collect_primitives_into_treelets(treelets);
}

__global__ void hlbvh_build_bottom_bvh(HLBVH *bvh, const BottomBVHArgs *bvh_args_array,
                                       uint array_length) {
    bvh->build_bottom_bvh(bvh_args_array, array_length);
}

__global__ void init_bvh_args(BottomBVHArgs *bvh_args_array, uint *accumulated_offset,
                              const BVHBuildNode *bvh_build_nodes, const uint start,
                              const uint end) {
    if (gridDim.x * gridDim.y * gridDim.z > 1) {
        printf("init_bvh_args(): launching more than 1 blocks destroys inter-thread "
               "synchronization.\n");
        asm("trap;");
    }

    const uint worker_idx = threadIdx.x;

    __shared__ cuda::atomic<uint, cuda::thread_scope_block> shared_accumulated_offset;
    if (worker_idx == 0) {
        shared_accumulated_offset = *accumulated_offset;
    }
    __syncthreads();

    const uint total_jobs = end - start;
    const uint jobs_per_worker = total_jobs / blockDim.x + 1;

    for (uint job_offset = 0; job_offset < jobs_per_worker; job_offset++) {
        const uint idx = worker_idx * jobs_per_worker + job_offset;
        if (idx >= total_jobs) {
            break;
        }

        const uint build_node_idx = idx + start;
        const auto &node = bvh_build_nodes[build_node_idx];

        if (!node.is_leaf() || node.num_primitives <= MAX_PRIMITIVES_NUM_IN_LEAF) {
            bvh_args_array[idx].expand_leaf = false;
            continue;
        }

        bvh_args_array[idx].expand_leaf = true;
        bvh_args_array[idx].build_node_idx = build_node_idx;
        bvh_args_array[idx].left_child_idx = shared_accumulated_offset.fetch_add(2);
        // 2 pointers: one for left and another right child
    }

    __syncthreads();
    if (worker_idx == 0) {
        *accumulated_offset = shared_accumulated_offset;
    }
    __syncthreads();
}

__global__ void init_triangles_from_mesh(Triangle *triangles, const TriangleMesh *mesh) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= mesh->triangle_num) {
        return;
    }

    triangles[worker_idx].init(worker_idx, mesh);
}

__global__ void parallel_render(Renderer *renderer, int num_samples) {
    const PerspectiveCamera *camera = renderer->camera;

    uint width = camera->camera_base.resolution.x;
    uint height = camera->camera_base.resolution.y;

    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    renderer->evaluate_pixel_sample(Point2i(x, y), num_samples);
}

__global__ void copy_gpu_pixels_to_rgb(const Renderer *renderer, RGB *output_rgb) {
    const PerspectiveCamera *camera = renderer->camera;

    uint width = camera->camera_base.resolution.x;
    uint height = camera->camera_base.resolution.y;

    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    uint flat_idx = x + y * width;
    output_rgb[flat_idx] = renderer->film->get_pixel_rgb(Point2i(x, y));
}

void writer_to_file(const std::string &filename, const RGB *pixels_rgb, const Point2i &resolution) {
    int width = resolution.x;
    int height = resolution.y;

    SRGBColorEncoding srgb_encoding;
    std::vector<unsigned char> png_pixels(width * height * 4);

    for (uint y = 0; y < height; y++) {
        for (uint x = 0; x < width; x++) {
            uint index = y * width + x;
            const auto rgb = pixels_rgb[index];

            if (rgb.has_nan()) {
                printf("writer_to_file(): pixel(%d, %d): has a NAN component\n", x, y);
            }

            png_pixels[4 * index + 0] = srgb_encoding.from_linear(rgb.r);
            png_pixels[4 * index + 1] = srgb_encoding.from_linear(rgb.g);
            png_pixels[4 * index + 2] = srgb_encoding.from_linear(rgb.b);
            png_pixels[4 * index + 3] = 255;
        }
    }

    // Encode the image
    // if there's an error, display it
    if (unsigned error = lodepng::encode(filename, png_pixels, width, height); error) {
        std::cerr << "lodepng::encoder error " << error << ": " << lodepng_error_text(error)
                  << std::endl;
        throw std::runtime_error("lodepng::encode() fail");
    }
}
} // namespace GPU

#pragma once

#include <iostream>
#include <string>

#include <curand_kernel.h>

#include "ext/lodepng/lodepng.h"

#include "pbrt/spectra/constants.h"
#include "pbrt/spectra/color_encoding.h"
#include "pbrt/spectra/piecewise_linear_spectrum.h"
#include "pbrt/spectra/rgb_color_space.h"
#include "pbrt/spectra/sampled_wavelengths.h"

#include "pbrt/accelerator/hlbvh.h"

#include "pbrt/base/shape.h"

#include "pbrt/filters/box.h"

#include "pbrt/films/pixel_sensor.h"
#include "pbrt/films/rgb_film.h"

#include "pbrt/cameras/perspective.h"

#include "pbrt/integrators/surface_normal.h"
#include "pbrt/integrators/ambient_occlusion.h"

#include "pbrt/samplers/independent.h"

namespace GPU {
class GlobalVariable {
  public:
    PBRT_GPU ~GlobalVariable() {
        delete rgb_color_space;
    }

    PBRT_GPU
    GlobalVariable(const double *cie_lambdas, const double *cie_x_value, const double *cie_y_value,
                   const double *cie_z_value, const double *cie_illum_d6500, int length_d65,
                   const RGBtoSpectrumData::RGBtoSpectrumTableGPU *rgb_to_spectrum_table,
                   RGBtoSpectrumData::Gamut gamut) {
        auto _cie_x_piecewise_linear_spectrum =
            PiecewiseLinearSpectrum(cie_lambdas, cie_x_value, NUM_CIE_SAMPLES);
        auto _cie_y_piecewise_linear_spectrum =
            PiecewiseLinearSpectrum(cie_lambdas, cie_y_value, NUM_CIE_SAMPLES);
        auto _cie_z_piecewise_linear_spectrum =
            PiecewiseLinearSpectrum(cie_lambdas, cie_z_value, NUM_CIE_SAMPLES);

        cie_x = DenselySampledSpectrum(_cie_x_piecewise_linear_spectrum);
        cie_y = DenselySampledSpectrum(_cie_y_piecewise_linear_spectrum);
        cie_z = DenselySampledSpectrum(_cie_z_piecewise_linear_spectrum);

        auto illum_d65 =
            PiecewiseLinearSpectrum::from_interleaved(cie_illum_d6500, length_d65, true, &cie_y);

        if (gamut == RGBtoSpectrumData::SRGB) {
            rgb_color_space = new RGBColorSpace(
                Point2f(0.64, 0.33), Point2f(0.3, 0.6), Point2f(0.15, 0.06), illum_d65,
                rgb_to_spectrum_table, std::array<const Spectrum *, 3>{&cie_x, &cie_y, &cie_z});

            return;
        }

        printf("\nthis color space is not implemented\n\n");
        asm("trap;");
    }

    PBRT_GPU std::array<const Spectrum *, 3> get_cie_xyz() const {
        return {&cie_x, &cie_y, &cie_z};
    }

    const RGBColorSpace *rgb_color_space = nullptr;

  private:
    DenselySampledSpectrum cie_x;
    DenselySampledSpectrum cie_y;
    DenselySampledSpectrum cie_z;
};

class Renderer {
  public:
    const Integrator *integrator = nullptr;
    const Camera *camera = nullptr;
    const Filter *filter = nullptr;
    Film *film = nullptr;
    HLBVH *bvh = nullptr;

    const GlobalVariable *global_variables = nullptr;

    PixelSensor sensor;

    PBRT_GPU ~Renderer() {
        delete integrator;
        delete camera;
        delete filter;
        delete film;
        delete global_variables;
    }

    PBRT_GPU void evaluate_pixel_sample(const Point2i &p_pixel, const int num_samples) {
        int width = camera->resolution.x;
        int pixel_index = p_pixel.y * width + p_pixel.x;

        auto sampler = IndependentSampler(pixel_index);

        for (int i = 0; i < num_samples; ++i) {
            auto camera_sample = sampler.get_camera_sample(p_pixel, filter);
            auto lu = sampler.get_1d();
            auto lambda = SampledWavelengths::sample_visible(lu);

            auto ray = camera->generate_ray(camera_sample);

            auto radiance_l = ray.weight * integrator->li(ray.ray, lambda, bvh, sampler);

            film->add_sample(p_pixel, radiance_l, lambda, camera_sample.filter_weight);
        }
    }
};

template <typename S>
__global__ void build_shapes(Shape *shapes, const S *concrete_shapes, int num) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num) {
        return;
    }

    shapes[worker_idx].init(&concrete_shapes[worker_idx]);
}

__global__ void free_renderer(Renderer *renderer) {
    renderer->~Renderer();
    // renderer was never new in device code
    // so you have to destruct it manually
}

template <typename T>
__global__ void apply_transform(T *data, const Transform transform, int length) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= length) {
        return;
    }

    data[idx] = transform(data[idx]);
}

__global__ void gpu_init_rgb_to_spectrum_table_coefficients(
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

    int max_component = blockIdx.x;
    int z = blockIdx.y;
    int y = blockIdx.z;

    int x = threadIdx.x;
    int c = threadIdx.y;

    int idx = (((max_component * resolution + z) * resolution + y) * resolution + x) * 3 + c;

    rgb_to_spectrum_data->coefficients[max_component][z][y][x][c] =
        rgb_to_spectrum_table_coefficients[idx];
}

__global__ void
gpu_init_rgb_to_spectrum_table_scale(RGBtoSpectrumData::RGBtoSpectrumTableGPU *rgb_to_spectrum_data,
                                     const double *rgb_to_spectrum_table_scale) {
    int idx = threadIdx.x;
    rgb_to_spectrum_data->z_nodes[idx] = rgb_to_spectrum_table_scale[idx];
}

__global__ void
gpu_init_global_variables(Renderer *renderer, const double *cie_lambdas, const double *cie_x_value,
                          const double *cie_y_value, const double *cie_z_value,
                          const double *cie_illum_d6500, int length_d65,
                          const RGBtoSpectrumData::RGBtoSpectrumTableGPU *rgb_to_spectrum_table) {
    *renderer = Renderer();
    // don't new anything in constructor Renderer()

    renderer->global_variables =
        new GlobalVariable(cie_lambdas, cie_x_value, cie_y_value, cie_z_value, cie_illum_d6500,
                           length_d65, rgb_to_spectrum_table, RGBtoSpectrumData::SRGB);
}

__global__ void gpu_init_integrator(Renderer *renderer) {

    auto illuminant_spectrum = renderer->global_variables->rgb_color_space->illuminant;
    auto cie_y = renderer->global_variables->get_cie_xyz()[1];

    auto illuminant_scale = 1.0 / illuminant_spectrum->to_photometric(*cie_y);

    /*
    renderer->integrator = new SurfaceNormalIntegrator(
        *(renderer->global_variables->rgb_color_space), renderer->sensor);
    */

    renderer->integrator = new AmbientOcclusionIntegrator(illuminant_spectrum, illuminant_scale);
}

__global__ void gpu_init_filter(Renderer *renderer) {
    renderer->filter = new BoxFilter(0.5);
}

__global__ void gpu_init_pixel_sensor_cie_1931(Renderer *renderer, const double *cie_s0,
                                               const double *cie_s1, const double *cie_s2,
                                               const double *cie_s_lambda) {
    // TODO: default value for PixelSensor, will remove later
    double iso = 100;
    double white_balance_val = 0.0;
    double exposure_time = 1.0;
    double imaging_ratio = exposure_time * iso / 100.0;

    auto d_illum =
        DenselySampledSpectrum::cie_d(white_balance_val == 0.0 ? 6500.0 : white_balance_val, cie_s0,
                                      cie_s1, cie_s2, cie_s_lambda);

    auto cie_xyz = renderer->global_variables->get_cie_xyz();
    const auto color_space = renderer->global_variables->rgb_color_space;

    renderer->sensor = PixelSensor::cie_1931(
        cie_xyz, color_space, white_balance_val == 0 ? nullptr : &d_illum, imaging_ratio);
}

__global__ void gpu_init_pixels(Pixel *pixels, Point2i dimension) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= dimension.x * dimension.y) {
        return;
    }

    pixels[idx] = Pixel();
}

__global__ void gpu_init_rgb_film(Renderer *renderer, Point2i dimension, Pixel *pixels) {
    renderer->film = new RGBFilm(pixels, &(renderer->sensor), dimension,
                                 renderer->global_variables->rgb_color_space);
}

__global__ void gpu_init_camera(Renderer *renderer, const Point2i resolution,
                                const CameraTransform camera_transform, const double fov) {
    renderer->camera = new PerspectiveCamera(resolution, camera_transform, fov, 0.0);
}

__global__ void hlbvh_init_morton_primitives(HLBVH *bvh) {
    bvh->init_morton_primitives();
}

__global__ void hlbvh_init_treelests(HLBVH *bvh, Treelet *treelets) {
    bvh->init_treelets(treelets);
}

__global__ void hlbvh_compute_morton_code(HLBVH *bvh, const Bounds3f bounds_of_centroids) {
    bvh->compute_morton_code(bounds_of_centroids);
}

__global__ void hlbvh_build_treelets(HLBVH *bvh, Treelet *treelets) {
    bvh->collect_primitives_into_treelets(treelets);
}

__global__ void hlbvh_build_bottom_bvh(HLBVH *bvh, const BVHArgs *bvh_args_array,
                                       uint array_length) {
    bvh->build_bottom_bvh(bvh_args_array, array_length);
}

__global__ void init_triangles_from_mesh(Triangle *triangles, const TriangleMesh *mesh) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= mesh->triangle_num) {
        return;
    }

    triangles[worker_idx].init(worker_idx, mesh);
}

__global__ void parallel_render(Renderer *renderer, int num_samples) {
    const Camera *camera = renderer->camera;

    int width = camera->resolution.x;
    int height = camera->resolution.y;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    renderer->evaluate_pixel_sample(Point2i(x, y), num_samples);
}

__global__ void copy_gpu_pixels_to_rgb(const Renderer *renderer, RGB *output_rgb) {
    const Camera *camera = renderer->camera;

    int width = camera->resolution.x;
    int height = camera->resolution.y;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) {
        return;
    }

    int flat_idx = x + y * width;
    output_rgb[flat_idx] = renderer->film->get_pixel_rgb(Point2i(x, y));
}

void writer_to_file(const std::string &filename, const RGB *pixels_rgb, const Point2i &resolution) {
    int width = resolution.x;
    int height = resolution.y;

    SRGBColorEncoding srgb_encoding;
    std::vector<unsigned char> png_pixels(width * height * 4);

    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            int index = y * width + x;
            auto rgb = pixels_rgb[index];

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

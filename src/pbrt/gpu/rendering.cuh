#pragma once

#include <iostream>
#include <string>
#include <assert.h>

#include <curand_kernel.h>

#include "ext/lodepng/lodepng.h"

#include "pbrt/spectra/constants.h"
#include "pbrt/spectra/color_encoding.h"
#include "pbrt/spectra/piecewise_linear_spectrum.h"
#include "pbrt/spectra/rgb_color_space.h"
#include "pbrt/spectra/sampled_wavelengths.h"

#include "pbrt/filters/box.h"
#include "pbrt/films/pixel_sensor.h"
#include "pbrt/films/rgb_film.h"
#include "pbrt/cameras/perspective.h"
#include "pbrt/shapes/triangle.h"
#include "pbrt/integrators/surface_normal.h"
#include "pbrt/integrators/ambient_occlusion.h"
#include "pbrt/samplers/independent.h"

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

    PBRT_GPU [[nodiscard]] std::array<const Spectrum *, 3> get_cie_xyz() const {
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
    Aggregate *aggregate = nullptr;
    const Filter *filter = nullptr;
    Film *film = nullptr;

    const GlobalVariable *global_varialbes = nullptr;

    Pixel *pixels = nullptr;
    PixelSensor sensor;

    PBRT_GPU ~Renderer() {
        delete integrator;
        delete aggregate;
        delete camera;
        delete filter;
        delete film;
        delete global_varialbes;
    }

    PBRT_GPU void evaluate_pixel_sample(const Point2i &p_pixel, const int num_samples) {
        int width = camera->resolution.x;
        int pixel_index = p_pixel.y * width + p_pixel.x;

        auto sampler = IndependentSampler(pixel_index);

        pixels[pixel_index] = Pixel();

        for (int i = 0; i < num_samples; ++i) {
            auto camera_sample = sampler.get_camera_sample(p_pixel, filter);
            auto lu = sampler.get_1d();
            auto lambda = SampledWavelengths::sample_visible(lu);

            auto ray = camera->generate_ray(camera_sample);

            auto radiance_l = ray.weight * integrator->li(ray.ray, lambda, aggregate, sampler);

            film->add_sample(p_pixel, radiance_l, lambda, camera_sample.filter_weight);
        }
    }
};

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

    renderer->global_varialbes =
        new GlobalVariable(cie_lambdas, cie_x_value, cie_y_value, cie_z_value, cie_illum_d6500,
                           length_d65, rgb_to_spectrum_table, RGBtoSpectrumData::SRGB);
}

__global__ void gpu_init_aggregate(Renderer *renderer) {
    renderer->aggregate = new Aggregate();
}

__global__ void gpu_init_integrator(Renderer *renderer) {

    auto illuminant_spectrum = renderer->global_varialbes->rgb_color_space->illuminant;
    auto cie_y = renderer->global_varialbes->get_cie_xyz()[1];

    auto illuminant_scale = 1.0 / illuminant_spectrum->to_photometric(*cie_y);

    /*
    renderer->integrator = new SurfaceNormalIntegrator(
        *(renderer->global_varialbes->rgb_color_space), renderer->sensor);
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

    auto cie_xyz = renderer->global_varialbes->get_cie_xyz();
    const auto color_space = renderer->global_varialbes->rgb_color_space;

    renderer->sensor = PixelSensor::cie_1931(
        cie_xyz, color_space, white_balance_val == 0 ? nullptr : &d_illum, imaging_ratio);
}

__global__ void gpu_init_rgb_film(Renderer *renderer, Point2i dimension) {
    if (renderer->pixels == nullptr) {
        printf("ERROR: renderer->pixels is nullptr\n\n");
        asm("trap;");
    }

    renderer->film = new RGBFilm(renderer->pixels, &(renderer->sensor), dimension,
                                 renderer->global_varialbes->rgb_color_space);
}

__global__ void gpu_init_camera(Renderer *renderer, const Point2i resolution,
                                const CameraTransform camera_transform, const double fov) {
    renderer->camera = new PerspectiveCamera(resolution, camera_transform, fov, 0.0);
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

__global__ void gpu_parallel_render(Renderer *renderer, int num_samples) {
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

__global__ void write_frame_buffer_to_rgb(const Renderer *renderer, RGB *output_rgb) {
    const Camera *camera = renderer->camera;

    int width = camera->resolution.x;
    int height = camera->resolution.y;

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= width * height) {
        return;
    }

    renderer->film->write_to_rgb(output_rgb, idx);
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

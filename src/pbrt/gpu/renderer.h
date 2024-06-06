#pragma once

#include <iostream>
#include <string>

#include "pbrt/base/camera.h"
#include "pbrt/base/filter.h"
#include "pbrt/base/integrator.h"
#include "pbrt/base/sampler.h"
#include "pbrt/base/shape.h"
#include "pbrt/base/spectrum.h"

#include "pbrt/accelerator/hlbvh.h"

#include "pbrt/cameras/perspective.h"

#include "pbrt/films/pixel_sensor.h"
#include "pbrt/films/rgb_film.h"

#include "pbrt/gpu/global_variable.h"

#include "pbrt/lights/diffuse_area_light.h"

#include "pbrt/primitives/simple_primitives.h"
#include "pbrt/primitives/geometric_primitive.h"

#include "pbrt/samplers/independent_sampler.h"

#include "pbrt/shapes/triangle.h"

#include "pbrt/spectrum_util/spectrum_constants.h"
#include "pbrt/spectrum_util/color_encoding.h"
#include "pbrt/spectrum_util/rgb_color_space.h"
#include "pbrt/spectrum_util/sampled_wavelengths.h"
#include "pbrt/spectra/densely_sampled_spectrum.h"

namespace GPU {

class Renderer {
  public:
    Integrator *integrator;
    Camera *camera;
    Filter *filter;
    Film *film;
    HLBVH *bvh;
    Sampler *samplers;

    const GlobalVariable *global_variables;

    PixelSensor sensor;

    PBRT_GPU void evaluate_pixel_sample(const Point2i p_pixel, const int num_samples) {
        int width = camera->get_camerabase()->resolution.x;
        const uint pixel_index = p_pixel.y * width + p_pixel.x;

        auto sampler = &samplers[pixel_index];
        sampler->start_pixel_sample(pixel_index, 0, 0);

        for (uint i = 0; i < num_samples; ++i) {
            auto camera_sample = sampler->get_camera_sample(p_pixel, filter);
            auto lu = sampler->get_1d();
            auto lambda = SampledWavelengths::sample_visible(lu);

            auto ray = camera->generate_camera_differential_ray(camera_sample);

            auto radiance_l = ray.weight * integrator->li(ray.ray, lambda, sampler);

            if (radiance_l.has_nan()) {
                printf("%s(): pixel(%d, %d), samples %u: has NAN\n", __func__, p_pixel.x, p_pixel.y,
                       i);
            }

            film->add_sample(p_pixel, radiance_l, lambda, camera_sample.filter_weight);
        }
    }
};

__global__ static void init_triangles_from_mesh(Triangle *triangles, const TriangleMesh *mesh) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= mesh->triangles_num) {
        return;
    }

    triangles[worker_idx].init(worker_idx, mesh);
}

template <typename TypeOfLight>
static __global__ void init_lights(Light *lights, TypeOfLight *concrete_shapes, uint num) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num) {
        return;
    }

    lights[worker_idx].init(&concrete_shapes[worker_idx]);
}

template <typename TypeOfShape>
static __global__ void init_shapes(Shape *shapes, const TypeOfShape *concrete_shapes, uint num) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num) {
        return;
    }

    shapes[worker_idx].init(&concrete_shapes[worker_idx]);
}

static __global__ void init_simple_primitives(SimplePrimitive *simple_primitives,
                                              const Shape *shapes, const Material *material,
                                              uint length) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= length) {
        return;
    }

    simple_primitives[idx].init(&shapes[idx], material);
}

static __global__ void init_geometric_primitives(GeometricPrimitive *geometric_primitives,
                                                 const Shape *shapes, const Material *material,
                                                 const Light *area_light, uint length) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= length) {
        return;
    }

    geometric_primitives[idx].init(&shapes[idx], material, &area_light[idx]);
}

template <typename TypeOfPrimitive>
static __global__ void init_primitives(Primitive *primitives, TypeOfPrimitive *_primitives,
                                       uint length) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= length) {
        return;
    }

    primitives[idx].init(&_primitives[idx]);
}

static __global__ void init_independent_samplers(IndependentSampler *samplers,
                                                 uint samples_per_pixel, uint length) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= length) {
        return;
    }

    samplers[idx].init(samples_per_pixel);
}

template <typename T>
static __global__ void init_samplers(Sampler *samplers, T *_samplers, uint length) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= length) {
        return;
    }

    samplers[idx].init(&_samplers[idx]);
}

template <typename T>
static __global__ void apply_transform(T *data, const Transform transform, uint length) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= length) {
        return;
    }

    data[idx] = transform(data[idx]);
}

static __global__ void init_pixels(Pixel *pixels, Point2i dimension) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= dimension.x * dimension.y) {
        return;
    }

    pixels[idx].init_zero();
}

__global__ static void parallel_render(Renderer *renderer, int num_samples) {
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
} // namespace GPU

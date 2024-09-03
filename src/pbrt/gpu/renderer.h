#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/util/macro.h"
#include <string>
#include <vector>

class Camera;
class Film;
class Filter;
class HLBVH;
class Integrator;
class Sampler;
class WavefrontPathIntegrator;

struct Renderer {
    static Renderer *create(std::vector<void *> &gpu_dynamic_pointers) {
        Renderer *renderer;
        CHECK_CUDA_ERROR(cudaMallocManaged(&renderer, sizeof(Renderer)));
        gpu_dynamic_pointers.push_back(renderer);

        renderer->megakernel_integrator = nullptr;
        renderer->wavefront_integrator = nullptr;
        renderer->camera = nullptr;
        renderer->film = nullptr;
        renderer->filter = nullptr;
        renderer->bvh = nullptr;
        renderer->samplers = nullptr;

        return renderer;
    }

    const Integrator *megakernel_integrator;
    WavefrontPathIntegrator *wavefront_integrator;
    Camera *camera;
    Film *film;
    const Filter *filter;
    const HLBVH *bvh;
    Sampler *samplers;

    PBRT_GPU
    void evaluate_pixel_sample(const Point2i p_pixel, const uint num_samples);

    void render(const std::string &output_filename, uint samples_per_pixel);
};

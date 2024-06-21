#pragma once

#include <string>
#include <vector>

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/util/macro.h"

class Camera;
class Film;
class Filter;
class HLBVH;
class Integrator;
class Sampler;

struct Renderer {
    static Renderer *create(std::vector<void *> &gpu_dynamic_pointers) {
        Renderer *renderer;
        CHECK_CUDA_ERROR(cudaMallocManaged(&renderer, sizeof(Renderer)));

        gpu_dynamic_pointers.push_back(renderer);

        return renderer;
    }

    const Integrator *integrator;
    Camera *camera;
    Film *film;
    const Filter *filter;
    const HLBVH *bvh;
    Sampler *samplers;

    PBRT_GPU
    void evaluate_pixel_sample(const Point2i p_pixel, const uint num_samples);

    void render(const std::string &output_filename, const Point2i &film_resolution,
                uint samples_per_pixel);
};

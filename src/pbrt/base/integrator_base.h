#pragma once

#include <pbrt/gpu/macro.h>
#include <pbrt/util/optional.h>

class Camera;
class Filter;
class HLBVH;
class Interaction;
class Light;
class UniformLightSampler;
class PowerLightSampler;
class Ray;
class SampledSpectrum;
struct ShapeIntersection;

struct IntegratorBase {
    const HLBVH *bvh;
    const Camera *camera;
    const Filter *filter;

    const Light **lights;
    int light_num;

    const Light **infinite_lights;
    int infinite_light_num;

    const PowerLightSampler *light_sampler;

    void init() {
        bvh = nullptr;
        camera = nullptr;
        filter = nullptr;
        light_sampler = nullptr;

        lights = nullptr;
        light_num = 0;

        infinite_lights = nullptr;
        infinite_light_num = 0;
    }

    [[nodiscard]] bool is_ready() const {
        if (bvh == nullptr || camera == nullptr || filter == nullptr || light_sampler == nullptr) {
            return false;
        }

        if (light_num > 0 && lights == nullptr) {
            return false;
        }

        if (infinite_light_num > 0 && infinite_lights == nullptr) {
            return false;
        }

        return true;
    }

    PBRT_CPU_GPU
    bool fast_intersect(const Ray &ray, Real t_max) const;

    PBRT_CPU_GPU
    bool unoccluded(const Interaction &p0, const Interaction &p1) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeIntersection> intersect(const Ray &ray, Real t_max) const;

    PBRT_CPU_GPU
    SampledSpectrum tr(const Interaction &p0, const Interaction &p1) const;
};

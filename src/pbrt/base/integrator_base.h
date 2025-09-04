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
class SampledWavelengths;

struct ShapeIntersection;

struct IntegratorBase {
    static constexpr Real interface_bounce_contribution = 0.3;

    IntegratorBase() = default;

    const HLBVH *bvh = nullptr;
    const Camera *camera = nullptr;
    const Filter *filter = nullptr;

    const Light **lights = nullptr;
    int light_num = 0;

    const Light **infinite_lights = nullptr;
    int infinite_light_num = 0;

    const PowerLightSampler *light_sampler = nullptr;

    [[nodiscard]]
    bool is_ready() const {
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
    [[nodiscard]]
    SampledSpectrum compute_transmittance(const Interaction &p0, const Interaction &p1,
                                          const SampledWavelengths &lambda, int max_depth) const;
};

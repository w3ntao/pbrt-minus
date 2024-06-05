#pragma once

#include "pbrt/util/macro.h"

class HLBVH;
class Camera;
class Light;
class UniformLightSampler;
class ImageInfiniteLight;

class Ray;
class Interaction;

struct IntegratorBase {
    const HLBVH *bvh;
    const Camera *camera;
    const Light **lights;
    const UniformLightSampler *uniform_light_sampler;
    uint light_num;

    const Light **infinite_lights;
    uint infinite_light_num;

    PBRT_GPU
    bool fast_intersect(const Ray &ray, FloatType t_max) const;

    PBRT_GPU
    bool unoccluded(const Interaction &p0, const Interaction &p1) const;
};

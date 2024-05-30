#pragma once

class HLBVH;
class Camera;
class Light;
class UniformLightSampler;
class ImageInfiniteLight;

struct IntegratorBase {
    const HLBVH *bvh;
    const Camera *camera;
    const Light **lights;
    const UniformLightSampler *uniform_light_sampler;
    uint light_num;

    const Light **infinite_lights;
    uint infinite_light_num;
};

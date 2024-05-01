#pragma once

class HLBVH;
class Camera;
class Light;
class UniformLightSampler;

struct IntegratorBase {
    const HLBVH *bvh;
    const Camera *camera;
    const Light **lights;
    const UniformLightSampler *uniform_light_sampler;
    uint light_num;
};

#pragma once

class HLBVH;
class Camera;

struct IntegratorBase {
    const HLBVH *bvh;
    const Camera *camera;
};

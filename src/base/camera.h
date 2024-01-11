#pragma once

#include "base/ray.h"

class Camera {
    public:
        __device__ virtual ~Camera() {}

        __device__ virtual Ray get_ray(float s, float t, curandState *local_rand_state) const = 0;
};

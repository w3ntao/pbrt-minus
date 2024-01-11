//
// Created by wentao on 4/11/23.
//

#ifndef CUDA_RAY_TRACER_CAMERA_H
#define CUDA_RAY_TRACER_CAMERA_H

#include "base/ray.h"

class Camera {
    public:
        __device__ virtual ~Camera() {}

        __device__ virtual Ray get_ray(float s, float t, curandState *local_rand_state) const = 0;
};

#endif // CUDA_RAY_TRACER_CAMERA_H

#pragma once

#include "base/ray.h"

class Camera {
  public:
    PBRT_GPU virtual ~Camera() {}

    PBRT_GPU virtual Ray get_ray(double s, double t, curandState *local_rand_state) const = 0;
};

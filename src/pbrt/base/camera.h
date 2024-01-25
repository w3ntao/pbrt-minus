#pragma once

#include "pbrt/base/ray.h"

class Camera {
  public:
    const int width = -1;
    const int height = -1;

    PBRT_GPU Camera(int _width, int _height) : width(_width), height(_height) {}

    PBRT_GPU virtual ~Camera() {}

    PBRT_GPU virtual Ray get_ray(double s, double t, curandState *local_rand_state) const = 0;
};

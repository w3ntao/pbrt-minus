#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/base/camera.h"

class Filter;

class Sampler {
  public:
    PBRT_GPU virtual ~Sampler() {}

    PBRT_GPU virtual double get_1d() = 0;

    PBRT_GPU virtual Point2f get_2d() = 0;

    PBRT_GPU virtual Point2f get_pixel_2d() = 0;

    PBRT_GPU CameraSample get_camera_sample(const Point2i &pPixel, const Filter *filter);
};

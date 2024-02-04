#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/base/camera.h"

class Sampler {
  public:
    PBRT_GPU virtual ~Sampler() {}

    PBRT_GPU virtual double get_1d() = 0;

    PBRT_GPU virtual Point2f get_2d() = 0;

    PBRT_GPU virtual Point2f get_pixel_2d() = 0;

    PBRT_GPU CameraSample get_camera_sample(const Point2i &pPixel, const Filter *filter) {
        auto fs = filter->sample(get_pixel_2d());

        return CameraSample(pPixel.to_point2f() + fs.p + Vector2f(0.5, 0.5), get_2d(), fs.weight);
    }
};

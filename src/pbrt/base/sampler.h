#pragma once

#include "pbrt/euclidean_space/point2.h"

class Sampler {
  public:
    PBRT_GPU virtual ~Sampler() {}

    PBRT_GPU virtual double get_1d() = 0;
    PBRT_GPU virtual Point2f get_2d() = 0;
};

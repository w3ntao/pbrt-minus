#pragma once

#include "pbrt/base/filter.h"
#include "pbrt/euclidean_space/point2.h"

class BoxFilter {
  public:
    PBRT_CPU_GPU void init(double _radius) {
        radius = Point2f(_radius, _radius);
    }

    PBRT_GPU
    FilterSample sample(Point2f u) const {
        Point2f p(lerp(u[0], -radius.x, radius.x), lerp(u[1], -radius.y, radius.y));
        return FilterSample(p, 1.0);
    }

  private:
    Point2f radius;
};

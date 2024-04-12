#pragma once

#include "pbrt/base/filter.h"
#include "pbrt/euclidean_space/point2.h"
#include "pbrt/util/utility_math.h"

class BoxFilter {
  public:
    void init(FloatType _radius) {
        radius = Point2f(_radius, _radius);
    }

    PBRT_CPU_GPU
    FilterSample sample(const Point2f u) const {
        Point2f p(lerp(u[0], -radius.x, radius.x), lerp(u[1], -radius.y, radius.y));
        return FilterSample(p, 1.0);
    }

  private:
    Point2f radius;
};

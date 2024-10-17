#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/util/basic_math.h"

class BoxFilter {
  public:
    void init(FloatType _radius) {
        radius = Vector2f(_radius, _radius);
    }

    PBRT_CPU_GPU
    FilterSample sample(const Point2f u) const {
        Point2f p(lerp(u[0], -radius.x, radius.x), lerp(u[1], -radius.y, radius.y));
        return FilterSample(p, 1.0);
    }

    PBRT_CPU_GPU
    Vector2f get_radius() const {
        return radius;
    }

  private:
    Vector2f radius;
};

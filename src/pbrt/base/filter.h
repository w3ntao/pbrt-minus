#pragma once

#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/point2.h"

class BoxFilter;

struct FilterSample {
    Point2f p;
    FloatType weight;

    PBRT_CPU_GPU FilterSample() {
        p = Point2f(NAN, NAN);
        weight = NAN;
    }

    PBRT_CPU_GPU FilterSample(const Point2f _p, FloatType _weight) : p(_p), weight(_weight) {}
};

class Filter {
  public:
    enum class Type {
        box,
    };

    void init(const BoxFilter *box_filter);

    PBRT_CPU_GPU
    FilterSample sample(Point2f u) const;

  private:
    Type type;
    const void *ptr;
};

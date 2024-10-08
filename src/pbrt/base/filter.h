#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/util/macro.h"
#include <vector>

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

    static const Filter *create_box_filter(FloatType radius,
                                           std::vector<void *> &gpu_dynamic_pointers);

    void init(const BoxFilter *box_filter);

    PBRT_CPU_GPU
    FilterSample sample(Point2f u) const;

  private:
    Type type;
    const void *ptr;
};

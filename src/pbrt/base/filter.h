#pragma once

#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/point2.h"

class BoxFilter;

struct FilterSample {
    Point2f p;
    double weight;

    PBRT_CPU_GPU FilterSample(const Point2f _p, double _weight) : p(_p), weight(_weight) {}
};

class Filter {
  public:
    void init(BoxFilter *box_filter);

    PBRT_CPU_GPU
    FilterSample sample(Point2f u) const;

  private:
    enum class FilterType { box };

    void *filter_ptr;
    FilterType filter_type;

    PBRT_CPU_GPU void report_error() const {
        printf("\nFilter: this type is not implemented\n");
#if defined(__CUDA_ARCH__)
        asm("trap;");
#else
        throw std::runtime_error("Filter: this type is not implemented\n");
#endif
    }
};

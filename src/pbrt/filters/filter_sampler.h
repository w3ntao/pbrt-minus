#pragma once

#include <pbrt/distribution/piecewise_constant_2d.h>

class Filter;
class GPUMemoryAllocator;

struct FilterSample {
    Point2f p = Point2f(NAN, NAN);
    Real weight = NAN;

    PBRT_CPU_GPU
    FilterSample() {}

    PBRT_CPU_GPU
    FilterSample(const Point2f _p, Real _weight) : p(_p), weight(_weight) {}
};

class FilterSampler {
  public:
    FilterSampler(const Filter &filter, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    FilterSample sample(Point2f u) const {
        Real pdf;
        Point2i pi;
        Point2f p = distrib.sample(u, &pdf, &pi);

        return FilterSample(p, f[pi] / pdf);
    }

  private:
    Array2D<Real> f;
    Bounds2f domain;
    PiecewiseConstant2D distrib;
};

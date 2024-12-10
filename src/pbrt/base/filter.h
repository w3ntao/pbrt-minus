#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/util/array_2d.h"
#include "pbrt/util/piecewise_constant_2d.h"
#include <vector>

class BoxFilter;
class MitchellFilter;

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
        mitchell,
    };

    static const Filter *create_box_filter(FloatType radius,
                                           std::vector<void *> &gpu_dynamic_pointers);

    static const Filter *create_mitchell_filter(const Vector2f &radius, FloatType b, FloatType c,
                                                std::vector<void *> &gpu_dynamic_pointers);

    void init(const BoxFilter *box_filter);

    void init(const MitchellFilter *mitchell_filter);

    PBRT_CPU_GPU
    Vector2f radius() const;

    PBRT_CPU_GPU
    FloatType get_integral() const;

    PBRT_CPU_GPU
    FloatType evaluate(const Point2f p) const;

    PBRT_CPU_GPU
    FilterSample sample(const Point2f u) const;

  private:
    Type type;
    const void *ptr;
};

class FilterSampler {
  public:
    static const FilterSampler *create(const Filter *filter,
                                       std::vector<void *> &gpu_dynamic_pointers);

    void init(const Filter *filter, std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU
    FilterSample sample(Point2f u) const {
        FloatType pdf;
        Point2i pi;
        Point2f p = distrib.sample(u, &pdf, &pi);

        return FilterSample(p, f[pi] / pdf);
    }

  private:
    Array2D<FloatType> f;
    Bounds2f domain;
    PiecewiseConstant2D distrib;
};

#pragma once

#include <pbrt/distribution/piecewise_constant_2d.h>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/util/array_2d.h>

class BoxFilter;
class GaussianFilter;
class GPUMemoryAllocator;
class MitchellFilter;
class TriangleFilter;
class ParameterDictionary;

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
        gaussian,
        mitchell,
        triangle,
    };

    static const Filter *create(const std::string &filter_type,
                                const ParameterDictionary &parameters,
                                GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Vector2f get_radius() const;

    PBRT_CPU_GPU
    FloatType get_integral() const;

    PBRT_CPU_GPU
    FloatType evaluate(const Point2f &p) const;

    PBRT_CPU_GPU
    FilterSample sample(const Point2f &u) const;

  private:
    Type type;
    const void *ptr;

    void init(const BoxFilter *box_filter);

    void init(const GaussianFilter *gaussian_filter);

    void init(const MitchellFilter *mitchell_filter);

    void init(const TriangleFilter *triangle_filter);
};

class FilterSampler {
  public:
    static const FilterSampler *create(const Filter *filter, GPUMemoryAllocator &allocator);

    void init(const Filter *filter, GPUMemoryAllocator &allocator);

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

#pragma once

#include <pbrt/euclidean_space/point2.h>
#include <pbrt/util/array_2d.h>

class BoxFilter;
class GaussianFilter;
class GPUMemoryAllocator;
class MitchellFilter;
class TriangleFilter;
class ParameterDictionary;
struct FilterSample;

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
    Real get_integral() const;

    PBRT_CPU_GPU
    Real evaluate(const Point2f &p) const;

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

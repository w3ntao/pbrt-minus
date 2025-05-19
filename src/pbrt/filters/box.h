#pragma once

#include <pbrt/euclidean_space/point2.h>
#include <pbrt/util/math.h>

class GPUMemoryAllocator;
class ParameterDictionary;
struct FilterSample;

class BoxFilter {
  public:
    static const BoxFilter *create(const ParameterDictionary &parameters,
                                   GPUMemoryAllocator &allocator);

    BoxFilter() : radius(Vector2f(0.5, 0.5)) {}

    PBRT_CPU_GPU
    Real get_integral() const {
        return 4 * radius.x * radius.y;
    }

    PBRT_CPU_GPU
    Real evaluate(const Point2f p) const {
        return std::abs(p.x) <= radius.x && std::abs(p.y) <= radius.y ? 1 : 0;
    }

    PBRT_CPU_GPU
    FilterSample sample(const Point2f u) const;

    PBRT_CPU_GPU
    Vector2f get_radius() const {
        return radius;
    }

  private:
    Vector2f radius;
};

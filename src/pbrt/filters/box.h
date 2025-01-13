#pragma once

#include <pbrt/base/filter.h>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/util/basic_math.h>

class GPUMemoryAllocator;

class BoxFilter {
  public:
    static const BoxFilter *create(const ParameterDictionary &parameters,
                                   GPUMemoryAllocator &allocator);

    BoxFilter() : radius(Vector2f(0.5, 0.5)) {}

    PBRT_CPU_GPU
    FloatType get_integral() const {
        return 4 * radius.x * radius.y;
    }

    PBRT_CPU_GPU
    FloatType evaluate(const Point2f p) const {
        return std::abs(p.x) <= radius.x && std::abs(p.y) <= radius.y ? 1 : 0;
    }

    PBRT_CPU_GPU
    FilterSample sample(const Point2f u) const {
        Point2f p(pbrt::lerp(u[0], -radius.x, radius.x), pbrt::lerp(u[1], -radius.y, radius.y));
        return FilterSample(p, 1.0);
    }

    PBRT_CPU_GPU
    Vector2f get_radius() const {
        return radius;
    }

  private:
    Vector2f radius;
};

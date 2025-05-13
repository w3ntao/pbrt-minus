#pragma once

#include <pbrt/base/filter.h>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/euclidean_space/vector2.h>

class GPUMemoryAllocator;
class ParameterDictionary;

class TriangleFilter {
  public:
    static const TriangleFilter *create(const ParameterDictionary &parameters,
                                        GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Vector2f get_radius() const {
        return radius;
    }

    PBRT_CPU_GPU
    Real get_integral() const {
        return sqr(radius.x) * sqr(radius.y);
    }

    PBRT_CPU_GPU
    Real evaluate(const Point2f &p) const {
        return std::max<Real>(0, radius.x - std::abs(p.x)) *
               std::max<Real>(0, radius.y - std::abs(p.y));
    }

    PBRT_CPU_GPU
    FilterSample sample(const Point2f &u) const;

  private:
    Vector2f radius;
};

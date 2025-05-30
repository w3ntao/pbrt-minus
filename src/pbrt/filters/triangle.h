#pragma once

#include <pbrt/euclidean_space/point2.h>
#include <pbrt/util/math.h>

class GPUMemoryAllocator;
class ParameterDictionary;
struct FilterSample;

class TriangleFilter {
  public:
    TriangleFilter(const ParameterDictionary &parameters);

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

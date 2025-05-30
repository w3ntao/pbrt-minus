#pragma once

#include <pbrt/euclidean_space/point2.h>

class FilterSampler;
class GPUMemoryAllocator;
class ParameterDictionary;
struct FilterSample;

class GaussianFilter {
  public:
    GaussianFilter(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU Vector2f get_radius() const {
        return radius;
    }

    PBRT_CPU_GPU
    Real evaluate(Point2f p) const;

    PBRT_CPU_GPU
    Real get_integral() const;

    PBRT_CPU_GPU
    FilterSample sample(Point2f u) const;

  private:
    Vector2f radius{NAN, NAN};
    Real sigma = NAN;
    Real expX = NAN;
    Real expY = NAN;

    const FilterSampler *sampler = nullptr;
};

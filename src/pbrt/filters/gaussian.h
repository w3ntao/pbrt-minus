#pragma once

#include <pbrt/base/filter.h>

class GPUMemoryAllocator;
class ParameterDictionary;

class GaussianFilter {
  public:
    static GaussianFilter *create(const ParameterDictionary &parameters,
                                  GPUMemoryAllocator &allocator);

    void init_sampler(const Filter *filter, GPUMemoryAllocator &allocator);

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
    void init(const Vector2f &_radius, Real _sigma = 0.5f);

    Vector2f radius;
    Real sigma, expX, expY;
    const FilterSampler *sampler;
};

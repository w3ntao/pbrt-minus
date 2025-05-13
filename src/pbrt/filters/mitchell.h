#pragma once

#include <pbrt/base/filter.h>

class GPUMemoryAllocator;
class ParameterDictionary;

class MitchellFilter {
  public:
    static MitchellFilter *create(const ParameterDictionary &parameters,
                                  GPUMemoryAllocator &allocator);

    void init_sampler(const Filter *filter, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Vector2f get_radius() const {
        return radius;
    }

    PBRT_CPU_GPU
    Real get_integral() const {
        return radius.x * radius.y / 4;
    }

    PBRT_CPU_GPU
    FilterSample sample(const Point2f u) const {
        return sampler->sample(u);
    }

    PBRT_CPU_GPU
    Real evaluate(const Point2f p) const {
        return mitchell_1d(2 * p.x / radius.x) * mitchell_1d(2 * p.y / radius.y);
    }

  private:
    Vector2f radius;
    Real b, c;
    const FilterSampler *sampler;

    PBRT_CPU_GPU
    Real mitchell_1d(Real x) const;

    void init(const Vector2f &_radius, Real _b, Real _c) {
        radius = _radius;
        b = _b;
        c = _c;

        sampler = nullptr;
    }
};

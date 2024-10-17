#pragma once

#include "pbrt/euclidean_space/vector2.h"
#include <vector>

class Filter;
class FilterSampler;

class MitchellFilter {
  public:
    void init(const Vector2f &_radius, FloatType _b, FloatType _c) {
        radius = _radius;
        b = _b;
        c = _c;

        sampler = nullptr;
    }

    void build_filter_sampler(const Filter *filter, std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU
    Vector2f get_radius() const {
        return radius;
    }

    PBRT_CPU_GPU
    FilterSample sample(const Point2f u) const {
        return sampler->sample(u);
    }

    PBRT_CPU_GPU
    FloatType evaluate(const Point2f p) const {
        return Mitchell1D(2 * p.x / radius.x) * Mitchell1D(2 * p.y / radius.y);
    }

  private:
    Vector2f radius;
    FloatType b, c;
    const FilterSampler *sampler;

    PBRT_CPU_GPU
    FloatType Mitchell1D(FloatType x) const;
};

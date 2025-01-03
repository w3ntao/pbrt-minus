#pragma once

#include "pbrt/base/filter.h"
#include <vector>

class ParameterDictionary;

class MitchellFilter {
  public:
    static MitchellFilter *create(const ParameterDictionary &parameters,
                                  std::vector<void *> &gpu_dynamic_pointers);

    void init_sampler(const Filter *filter, std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU
    Vector2f get_radius() const {
        return radius;
    }

    PBRT_CPU_GPU
    FloatType get_integral() const {
        return radius.x * radius.y / 4;
    }

    PBRT_CPU_GPU
    FilterSample sample(const Point2f u) const {
        return sampler->sample(u);
    }

    PBRT_CPU_GPU
    FloatType evaluate(const Point2f p) const {
        return mitchell_1d(2 * p.x / radius.x) * mitchell_1d(2 * p.y / radius.y);
    }

  private:
    Vector2f radius;
    FloatType b, c;
    const FilterSampler *sampler;

    PBRT_CPU_GPU
    FloatType mitchell_1d(FloatType x) const;

    void init(const Vector2f &_radius, FloatType _b, FloatType _c) {
        radius = _radius;
        b = _b;
        c = _c;

        sampler = nullptr;
    }
};

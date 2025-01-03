#pragma once

#include "pbrt/base/filter.h"
#include <vector>

class ParameterDictionary;

class GaussianFilter {
  public:
    static GaussianFilter *create(const ParameterDictionary &parameters,
                                  std::vector<void *> &gpu_dynamic_pointers);

    void init_sampler(const Filter *filter, std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU Vector2f get_radius() const {
        return radius;
    }

    PBRT_CPU_GPU
    FloatType evaluate(Point2f p) const;

    PBRT_CPU_GPU
    FloatType get_integral() const;

    PBRT_CPU_GPU
    FilterSample sample(Point2f u) const;

  private:
    void init(const Vector2f &_radius, FloatType _sigma = 0.5f);

    Vector2f radius;
    FloatType sigma, expX, expY;
    const FilterSampler *sampler;
};

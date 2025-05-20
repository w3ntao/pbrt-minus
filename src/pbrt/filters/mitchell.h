#pragma once

class FilterSampler;
class GPUMemoryAllocator;
class ParameterDictionary;
struct FilterSample;

class MitchellFilter {
  public:
    MitchellFilter(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Vector2f get_radius() const {
        return radius;
    }

    PBRT_CPU_GPU
    Real get_integral() const {
        return radius.x * radius.y / 4;
    }

    PBRT_CPU_GPU
    FilterSample sample(const Point2f u) const;

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
};

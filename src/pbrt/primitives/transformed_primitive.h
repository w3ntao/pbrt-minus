#pragma once

#include "pbrt/euclidean_space/transform.h"

class Primitive;

class TransformedPrimitive {
  public:
    PBRT_CPU_GPU
    void init(const Primitive *_primitive, const Transform _render_from_primitive) {
        primitive = _primitive;
        render_from_pritimive = _render_from_primitive;
    }

    PBRT_CPU_GPU
    Bounds3f bounds() const;

    PBRT_GPU
    bool fast_intersect(const Ray &ray, FloatType t_max) const;

    PBRT_GPU
    cuda::std::optional<ShapeIntersection> intersect(const Ray &ray, FloatType t_max) const;

  private:
    Transform render_from_pritimive;
    const Primitive *primitive;
};

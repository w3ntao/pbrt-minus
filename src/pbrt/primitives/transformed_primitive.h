#pragma once

#include <pbrt/euclidean_space/transform.h>

class Primitive;

class TransformedPrimitive {
  public:
    PBRT_CPU_GPU
    void init(const Primitive *_primitive, const Transform _render_from_primitive) {
        primitive = _primitive;
        render_from_pritimive = _render_from_primitive;
    }

    PBRT_CPU_GPU
    const Material *get_material() const;

    PBRT_CPU_GPU
    const Primitive *get_primitive() const {
        return primitive;
    }

    PBRT_CPU_GPU
    Bounds3f bounds() const;

    PBRT_CPU_GPU
    bool fast_intersect(const Ray &ray, Real t_max) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeIntersection> intersect(const Ray &ray, Real t_max) const;

  private:
    Transform render_from_pritimive;
    const Primitive *primitive;
};

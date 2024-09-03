#pragma once

#include "pbrt/base/material.h"
#include "pbrt/base/shape.h"

class SimplePrimitive {
  public:
    PBRT_CPU_GPU
    void init(const Shape *_shape, const Material *_material) {
        shape = _shape;
        material = _material;
    }

    PBRT_CPU_GPU
    const Material *get_material() const {
        return material;
    }

    PBRT_CPU_GPU
    Bounds3f bounds() const {
        return shape->bounds();
    }

    PBRT_GPU
    bool fast_intersect(const Ray &ray, FloatType t_max) const {
        return shape->fast_intersect(ray, t_max);
    }

    PBRT_GPU
    cuda::std::optional<ShapeIntersection> intersect(const Ray &ray, FloatType t_max) const {
        auto si = shape->intersect(ray, t_max);
        if (!si.has_value()) {
            return {};
        }

        si->interaction.set_intersection_properties(material, nullptr);
        return si;
    }

  private:
    const Shape *shape;
    const Material *material;
};

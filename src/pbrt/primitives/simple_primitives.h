#pragma once

#include "pbrt/base/shape.h"
#include "pbrt/base/material.h"

class SimplePrimitive {
  public:
    PBRT_CPU_GPU
    void init(const Shape *_shape_ptr, const Material *_material) {
        shape_ptr = _shape_ptr;
        material = _material;
    }

    PBRT_CPU_GPU
    Bounds3f bounds() const {
        return shape_ptr->bounds();
    }

    PBRT_GPU
    bool fast_intersect(const Ray &ray, FloatType t_max) const {
        return shape_ptr->fast_intersect(ray, t_max);
    }

    PBRT_GPU
    std::optional<ShapeIntersection> intersect(const Ray &ray, FloatType t_max) const {
        auto si = shape_ptr->intersect(ray, t_max);
        if (!si.has_value()) {
            return {};
        }

        si->interaction.set_intersection_properties(material, nullptr);
        return si;
    }

  private:
    const Shape *shape_ptr;
    const Material *material;
};

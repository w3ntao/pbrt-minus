#pragma once

#include <pbrt/base/shape.h>

class Material;

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

    PBRT_CPU_GPU
    bool fast_intersect(const Ray &ray, Real t_max) const {
        return shape->fast_intersect(ray, t_max);
    }

    PBRT_CPU_GPU
    pbrt::optional<ShapeIntersection> intersect(const Ray &ray, Real t_max) const {
        auto si = shape->intersect(ray, t_max);
        if (!si.has_value()) {
            return {};
        }

        si->interaction.set_intersection_properties(material, nullptr, {}, ray.medium);
        return si;
    }

  private:
    const Shape *shape = nullptr;
    const Material *material = nullptr;
};

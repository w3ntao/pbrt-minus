#pragma once

#include <pbrt/base/interaction.h>
#include <pbrt/euclidean_space/bounds3.h>
#include <pbrt/gpu/macro.h>

class Light;
class Material;
class Shape;

class GeometricPrimitive {
  public:
    PBRT_CPU_GPU
    void init(const Shape *_shape_ptr, const Material *_material, const Light *_area_light);

    PBRT_CPU_GPU
    const Material *get_material() const;

    PBRT_CPU_GPU
    Bounds3f bounds() const;

    PBRT_CPU_GPU
    bool fast_intersect(const Ray &ray, Real t_max) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeIntersection> intersect(const Ray &ray, Real t_max) const;

  private:
    const Shape *shape_ptr;
    const Material *material;
    const Light *area_light;
};

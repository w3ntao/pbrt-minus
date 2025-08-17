#pragma once

#include <pbrt/base/interaction.h>
#include <pbrt/euclidean_space/bounds3.h>
#include <pbrt/gpu/macro.h>

class Light;
class Material;
class Shape;

struct MediumInterface;

class GeometricPrimitive {
  public:
    PBRT_CPU_GPU
    GeometricPrimitive::GeometricPrimitive(const Shape *_shape_ptr, const Material *_material,
                                           const Light *_area_light,
                                           const MediumInterface *_medium_interface)
        : shape_ptr(_shape_ptr), material(_material), area_light(_area_light),
          medium_interface(_medium_interface) {}

    PBRT_CPU_GPU
    const Material *get_material() const;

    PBRT_CPU_GPU
    Bounds3f bounds() const;

    PBRT_CPU_GPU
    bool fast_intersect(const Ray &ray, Real t_max) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeIntersection> intersect(const Ray &ray, Real t_max) const;

  private:
    const Shape *shape_ptr = nullptr;
    const Material *material = nullptr;
    const Light *area_light = nullptr;

    const MediumInterface *medium_interface = nullptr;
};

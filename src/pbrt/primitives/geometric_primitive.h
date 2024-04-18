#pragma once

#include <optional>

#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/base/interaction.h"

class Shape;
class DiffuseMaterial;
class DiffuseAreaLight;

class GeometricPrimitive {
  public:
    PBRT_CPU_GPU
    void init(const Shape *_shape_ptr, const DiffuseMaterial *_diffuse_material,
              const DiffuseAreaLight *_area_light);

    PBRT_CPU_GPU
    Bounds3f bounds() const;

    PBRT_GPU
    bool fast_intersect(const Ray &ray, FloatType t_max) const;

    PBRT_GPU
    std::optional<ShapeIntersection> intersect(const Ray &ray, FloatType t_max) const;

  private:
    const Shape *shape_ptr;

    // TODO: progress 2024/04/19: generalize this material
    const DiffuseMaterial *diffuse_material;
    const DiffuseAreaLight *area_light;
};

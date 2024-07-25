#pragma once

#include <cuda/std/optional>

#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/base/interaction.h"

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

    PBRT_GPU
    bool fast_intersect(const Ray &ray, FloatType t_max) const;

    PBRT_GPU
    cuda::std::optional<ShapeIntersection> intersect(const Ray &ray, FloatType t_max) const;

  private:
    const Shape *shape_ptr;
    const Material *material;
    const Light *area_light;
};

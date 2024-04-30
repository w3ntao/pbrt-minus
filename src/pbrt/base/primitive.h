#pragma once

#include <optional>
#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/bounds3.h"
#include "interaction.h"

class SimplePrimitive;
class GeometricPrimitive;

class Primitive {
  public:
    enum class Type {
        simple_primitive,
        geometric_primitive,
    };

    PBRT_CPU_GPU
    void init(const SimplePrimitive *simple_primitive);

    PBRT_CPU_GPU
    void init(const GeometricPrimitive *geometric_primitive);

    PBRT_CPU_GPU
    Bounds3f bounds() const;

    PBRT_GPU
    bool fast_intersect(const Ray &ray, FloatType t_max) const;

    PBRT_GPU
    std::optional<ShapeIntersection> intersect(const Ray &ray, FloatType t_max) const;

  private:
    Type primitive_type;
    const void *primitive_ptr;
};

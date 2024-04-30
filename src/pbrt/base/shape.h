#pragma once

#include <optional>

#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/base/interaction.h"
#include "pbrt/base/ray.h"

class Triangle;

class Shape {
  public:
    enum class Type {
        triangle,
    };

    PBRT_CPU_GPU
    void init(const Triangle *triangle);

    PBRT_CPU_GPU
    Bounds3f bounds() const;

    PBRT_CPU_GPU
    FloatType area() const;

    PBRT_GPU
    bool fast_intersect(const Ray &ray, FloatType t_max) const;

    PBRT_GPU
    std::optional<ShapeIntersection> intersect(const Ray &ray, FloatType t_max) const;

  private:
    Type shape_type;
    void *shape_ptr;
};

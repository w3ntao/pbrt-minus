#pragma once

#include <optional>

#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/base/interaction.h"
#include "pbrt/base/ray.h"

class Triangle;

class Shape {
  public:
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
    enum class ShapeType { triangle };

    ShapeType shape_type;
    void *shape_ptr;

    PBRT_CPU_GPU void report_error() const {
        printf("\nShape: this type is not implemented\n");
#if defined(__CUDA_ARCH__)
        asm("trap;");
#else
        throw std::runtime_error("Shape: this type is not implemented\n");
#endif
    }
};

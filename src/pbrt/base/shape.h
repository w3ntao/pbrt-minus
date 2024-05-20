#pragma once

#include <cuda/std/optional>

#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/base/interaction.h"
#include "pbrt/base/ray.h"

class Triangle;

struct ShapeSampleContext {
    Point3fi pi;
    Normal3f n;
    Normal3f ns;

    PBRT_CPU_GPU
    Point3f p() const {
        return pi.to_point3f();
    }
};

struct ShapeSample {
    Interaction interaction;
    FloatType pdf;
};

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
    cuda::std::optional<ShapeIntersection> intersect(const Ray &ray, FloatType t_max) const;

    PBRT_GPU
    cuda::std::optional<ShapeSample> sample(const ShapeSampleContext &ctx, const Point2f u) const;

  private:
    Type type;
    const void *ptr;
};

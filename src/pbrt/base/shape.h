#pragma once

#include <optional>

#include "pbrt/base/ray.h"
#include "pbrt/base/interaction.h"
#include "pbrt/util/float.h"

// ShapeIntersection Definition
struct ShapeIntersection {
    SurfaceInteraction interation;
    double t_hit;

    PBRT_CPU_GPU ShapeIntersection(const SurfaceInteraction &si, double t)
        : interation(si), t_hit(t) {}
};

class Shape {
  public:
    PBRT_GPU virtual ~Shape() {}

    PBRT_GPU virtual std::optional<ShapeIntersection> intersect(const Ray &ray,
                                                                double t_max = Infinity) const = 0;
};

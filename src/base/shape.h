#pragma once

#include <optional>

#include "base/ray.h"
#include "base/interaction.h"
#include "euclidean_space/point2.h"
#include "util/float.h"

// ShapeIntersection Definition
struct ShapeIntersection {
    SurfaceInteraction interation;
    double t_hit;

    PBRT_CPU_GPU ShapeIntersection(const SurfaceInteraction &si, double t)
        : interation(si), t_hit(t) {}
};

class Material;

// TODO: delete (the old) Intersection
struct Intersection {
    double t;
    Point3f p;
    Vector3f n;
    const Material *material_ptr;
};

class Shape {
  public:
    PBRT_GPU virtual ~Shape() {}

    PBRT_GPU virtual std::optional<ShapeIntersection> intersect(const Ray &ray,
                                                                double t_max = Infinity) const = 0;

    PBRT_GPU virtual const Material *get_material_ptr() const = 0;
};

#pragma once

#include "base/ray.h"

class Material;

struct Intersection {
        double t;
        Point p;
        Vector3 n;
        const Material *material_ptr;
};

class Shape {
    public:
        PBRT_GPU virtual ~Shape() {}

        PBRT_GPU virtual bool intersect(Intersection &intersection, const Ray &ray, double t_min,
                                        double t_max) const = 0;

        PBRT_GPU virtual const Material *get_material_ptr() const = 0;
};

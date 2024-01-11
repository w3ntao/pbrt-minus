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
        __device__ virtual ~Shape() {}

        __device__ virtual bool intersect(Intersection &intersection, const Ray &ray, double t_min,
                                          double t_max) const = 0;

        __device__ virtual const Material *get_material_ptr() const = 0;
};

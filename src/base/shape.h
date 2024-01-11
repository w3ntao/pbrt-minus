#pragma once

#include "base/ray.h"

class Material;

struct Intersection {
        float t;
        Point p;
        Vector3 n;
        const Material *material_ptr;
};

class Shape {
    public:
        __device__ virtual ~Shape() {}

        __device__ virtual bool intersect(Intersection &intersection, const Ray &ray, float t_min,
                                          float t_max) const = 0;

        __device__ virtual const Material *get_material_ptr() const = 0;
};

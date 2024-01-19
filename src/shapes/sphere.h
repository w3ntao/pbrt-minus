#pragma once

#include "base/shape.h"

class Sphere : public Shape {
    Point3f center;
    double radius;
    const Material *material_ptr;

  public:
    PBRT_GPU Sphere(const Point3f &_center, double _radius, const Material *_material_ptr)
        : center(_center), radius(_radius), material_ptr(_material_ptr) {}

    PBRT_GPU ~Sphere() override {
        delete material_ptr;
    }

    PBRT_GPU std::optional<ShapeIntersection> intersect(const Ray &ray,
                                                        double t_max = Infinity) const override {

        // TODO: progress 2024/01/17 implementing Shape::intersect()
        return {};
    }

    PBRT_GPU const Material *get_material_ptr() const override {
        return material_ptr;
    }
};

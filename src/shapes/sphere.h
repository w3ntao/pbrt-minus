#pragma once

#include "base/shape.h"

class Sphere : public Shape {
    private:
        Point3f center;
        double radius;
        const Material *material_ptr;

    public:
        ~Sphere() override = default;

        PBRT_GPU Sphere(const Point3f &_center, double _radius, const Material *_material_ptr)
            : center(_center), radius(_radius), material_ptr(_material_ptr) {}

        PBRT_GPU bool intersect(Intersection &intersection, const Ray &ray, double t_min,
                                double t_max) const override {
            Vector3f oc = ray.o - center;
            double a = ray.d.dot(ray.d);
            double b = oc.dot(ray.d);
            double c = oc.dot(oc) - radius * radius;
            double discriminant = b * b - a * c;

            if (discriminant < 0.0) {
                return false;
            }

            double temp = (-b - sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                intersection.t = temp;
                intersection.p = ray.at(intersection.t);
                intersection.n = (intersection.p - center) / radius;
                intersection.material_ptr = material_ptr;
                return true;
            }

            temp = (-b + sqrt(discriminant)) / a;
            if (temp < t_max && temp > t_min) {
                intersection.t = temp;
                intersection.p = ray.at(intersection.t);
                intersection.n = (intersection.p - center) / radius;
                intersection.material_ptr = material_ptr;
                return true;
            }
        }

        PBRT_GPU const Material *get_material_ptr() const override {
            return material_ptr;
        }
};

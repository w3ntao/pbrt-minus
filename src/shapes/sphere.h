#pragma once

#include "base/shape.h"

class Sphere : public Shape {
    private:
        Point center;
        double radius;
        const Material *material_ptr;

    public:
        ~Sphere() override = default;

        __device__ Sphere(const Point &_center, double _radius, const Material *_material_ptr)
            : center(_center), radius(_radius), material_ptr(_material_ptr) {}

        __device__ bool intersect(Intersection &intersection, const Ray &ray, double t_min,
                                  double t_max) const override {
            Vector3 oc = ray.o - center;
            double a = dot(ray.d, ray.d);
            double b = dot(oc, ray.d);
            double c = dot(oc, oc) - radius * radius;
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

        __device__ const Material *get_material_ptr() const override {
            return material_ptr;
        }
};

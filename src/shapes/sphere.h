//
// Created by wentao on 4/7/23.
//

#ifndef CUDA_RAY_TRACER_SPHERE_H
#define CUDA_RAY_TRACER_SPHERE_H

#include "base/shape.h"

class Sphere : public Shape {
    private:
        Point center;
        float radius;
        const Material *material_ptr;

    public:
        ~Sphere() override = default;

        __device__ Sphere(const Point &_center, float _radius, const Material *_material_ptr)
            : center(_center), radius(_radius), material_ptr(_material_ptr) {}

        __device__ bool intersect(Intersection &intersection, const Ray &ray, float t_min,
                                  float t_max) const override {
            Vector3 oc = ray.o - center;
            float a = dot(ray.d, ray.d);
            float b = dot(oc, ray.d);
            float c = dot(oc, oc) - radius * radius;
            float discriminant = b * b - a * c;

            if (discriminant < 0.0) {
                return false;
            }

            float temp = (-b - sqrt(discriminant)) / a;
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

#endif // CUDA_RAY_TRACER_SPHERE_H

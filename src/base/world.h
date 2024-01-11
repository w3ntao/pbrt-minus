#pragma once

#include "base/shape.h"

class World {
    public:
        Shape **list;
        int size;

        PBRT_GPU World(Shape **_list, int n) : list(_list), size(n) {}

        PBRT_GPU bool intersect(Intersection &intersection, const Ray &ray, double t_min,
                                double t_max) const {
            bool intersected = false;
            double closest_so_far = t_max;
            Intersection temp_intersection;
            for (int idx = 0; idx < size; idx++) {
                if (list[idx]->intersect(temp_intersection, ray, t_min, closest_so_far)) {
                    intersected = true;
                    closest_so_far = temp_intersection.t;
                    intersection = temp_intersection;
                }
            }

            return intersected;
        }
};

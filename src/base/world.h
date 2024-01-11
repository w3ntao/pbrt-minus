//
// Created by wentao on 4/7/23.
//

#ifndef CUDA_RAY_TRACER_WORLD_H
#define CUDA_RAY_TRACER_WORLD_H

#include "base/shape.h"

class World {
    public:
        Shape **list;
        int size;

        __device__ World(Shape **_list, int n) : list(_list), size(n) {}
        
        __device__ bool intersect(Intersection &intersection, const Ray &ray, float t_min,
                                  float t_max) const {
            bool intersected = false;
            float closest_so_far = t_max;
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

#endif // CUDA_RAY_TRACER_WORLD_H

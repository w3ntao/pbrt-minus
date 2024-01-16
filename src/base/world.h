#pragma once

#include "base/shape.h"
#include "shapes/triangle.h"

class World {
    public:
        Shape **shapes;
        int shape_num;
        int current_num;

        PBRT_GPU World(int n) : current_num(0), shape_num(n) {
            shapes = new Shape *[n];
        }

        PBRT_GPU ~World() {
            for (int i = 0; i < current_num; ++i) {
                delete shapes[i];
            }
            delete[] shapes;
        }

        PBRT_GPU void add_triangles(const TriangleMesh *mesh) {
            // TODO:progress 2024/01/16 recovering

            for (int i = 0; i < mesh->triangle_num; ++i) {
                shapes[current_num] = new Triangle(i * 3, mesh);
                current_num += 1;
            }
        }

        PBRT_GPU bool intersect(Intersection &intersection, const Ray &ray, double t_min,
                                double t_max) const {
            bool intersected = false;
            double closest_so_far = t_max;
            Intersection temp_intersection;
            for (int idx = 0; idx < shape_num; idx++) {
                if (shapes[idx]->intersect(temp_intersection, ray, t_min, closest_so_far)) {
                    intersected = true;
                    closest_so_far = temp_intersection.t;
                    intersection = temp_intersection;
                }
            }

            return intersected;
        }
};
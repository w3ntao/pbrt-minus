#pragma once

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
        for (int i = 0; i < mesh->triangle_num; ++i) {
            shapes[current_num] = new Triangle(i, mesh);
            current_num += 1;
        }
    }

    PBRT_GPU std::optional<ShapeIntersection> intersect(const Ray &ray) const {
        double best_t = Infinity;
        std::optional<ShapeIntersection> best_intersection = {};
        for (int idx = 0; idx < shape_num; idx++) {
            const auto shape_intersection = shapes[idx]->intersect(ray);

            if (!shape_intersection || shape_intersection->t_hit > best_t) {
                continue;
            }

            best_t = shape_intersection->t_hit;
            best_intersection = shape_intersection;
        }

        return best_intersection;
    }
};

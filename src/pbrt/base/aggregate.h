#pragma once

#include "pbrt/util/dynamic_array.h"
#include "pbrt/shapes/triangle.h"
#include "pbrt/accelerator/bvh.h"

class Aggregate {
  private:
    BVH *bvh = nullptr;
    DynamicArray<const Shape *> shapes;

  public:
    PBRT_GPU ~Aggregate() {
        for (int idx = 0; idx < shapes.size(); idx++) {
            delete shapes[idx];
        }

        delete bvh;
    }

    PBRT_GPU void add_triangles(const TriangleMesh *mesh) {
        for (int i = 0; i < mesh->triangle_num; ++i) {
            auto triangle = new Triangle(i, mesh);
            shapes.push(triangle);
        }
    }

    PBRT_GPU void preprocess() {
        bvh = new BVH(shapes);
    }

    PBRT_GPU bool fast_intersect(const Ray &ray, double t_max) const {
        return bvh->fast_intersect(ray, t_max);
    }

    PBRT_GPU std::optional<ShapeIntersection> intersect(const Ray &ray) const {
        return bvh->intersect(ray, Infinity);
    }
};

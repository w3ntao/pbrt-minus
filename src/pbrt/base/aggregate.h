#pragma once

#include "pbrt/util/dynamic_array.h"
#include "pbrt/shapes/triangle.h"

class Aggregate {
  private:
    DynamicArray<const Shape *> shapes;

  public:
    HLBVH *hlbvh = nullptr;
    // TODO: decouple hlbvh from Aggregate

    PBRT_GPU ~Aggregate() {
        // TODO: manage Shape* in builder
        for (int idx = 0; idx < shapes.size(); idx++) {
            delete shapes[idx];
        }
    }

    PBRT_GPU void get_shape_num(int *num) const {
        *num = shapes.size();
    }

    PBRT_GPU void add_triangles(const TriangleMesh *mesh) {
        shapes.reserve(shapes.size() + mesh->triangle_num);

        for (int i = 0; i < mesh->triangle_num; ++i) {
            shapes.push(new Triangle(i, mesh));
        }
    }

    PBRT_GPU void init_hlbvh(BVHPrimitive *bvh_primitives, MortonPrimitive *morton_primitives) {
        hlbvh->init(shapes.data(), bvh_primitives, morton_primitives, shapes.size());
    }

    PBRT_GPU bool fast_intersect(const Ray &ray, double t_max) const {
        return hlbvh->fast_intersect(ray, t_max);
    }

    PBRT_GPU std::optional<ShapeIntersection> intersect(const Ray &ray) const {
        return hlbvh->intersect(ray, Infinity);
    }
};

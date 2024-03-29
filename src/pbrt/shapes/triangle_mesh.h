#pragma once

#include "pbrt/euclidean_space/transform.h"

class TriangleMesh {
  public:
    int triangle_num = -1;
    int points_num = -1;

    const int *vertex_indices = nullptr;
    const Point3f *p = nullptr;
    const Normal3f *n = nullptr;
    const Vector3f *s = nullptr;
    const Point2f *uv = nullptr;
    const int *faceIndices = nullptr;

    bool reverse_orientation = false;
    bool transformSwapsHandedness = false;

    PBRT_CPU_GPU
    void init(bool _reverse_orientation, const int *_vertex_indices, int num_indices,
              const Point3f *_points, int num_points) {
        reverse_orientation = _reverse_orientation;
        triangle_num = num_indices / 3;
        points_num = num_points;
        vertex_indices = _vertex_indices;
        p = _points;

        transformSwapsHandedness = false;
    }
};

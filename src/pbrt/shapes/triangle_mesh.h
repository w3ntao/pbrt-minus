#pragma once

#include "pbrt/euclidean_space/transform.h"

class TriangleMesh {
  public:
    uint triangles_num = 0;

    const int *vertex_indices;
    const Point3f *p;
    const Normal3f *n;
    const Vector3f *s;
    const Point2f *uv = nullptr;
    const int *faceIndices = nullptr;

    bool reverse_orientation;
    bool transformSwapsHandedness;

    PBRT_CPU_GPU
    void init(bool _reverse_orientation, const int *_vertex_indices, uint num_indices,
              const Point3f *_p, const Point2f *_uv) {
        reverse_orientation = _reverse_orientation;

        triangles_num = num_indices / 3;
        vertex_indices = _vertex_indices;
        p = _p;
        uv = _uv;

        // default value:
        n = nullptr;
        s = nullptr;

        transformSwapsHandedness = false;
    }
};

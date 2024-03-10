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

    PBRT_CPU_GPU TriangleMesh(bool _reverse_orientation, const int *_vertex_indices,
                              int num_indices, const Point3f *_points, int num_points)
        : reverse_orientation(_reverse_orientation), triangle_num(num_indices / 3),
          points_num(num_points), vertex_indices(_vertex_indices), p(_points) {}

    PBRT_CPU_GPU void operator=(const TriangleMesh &mesh) {
        triangle_num = mesh.triangle_num;
        points_num = mesh.points_num;
        vertex_indices = mesh.vertex_indices;
        p = mesh.p;
        n = mesh.n;
        s = mesh.s;
        uv = mesh.uv;
        faceIndices = mesh.faceIndices;
        reverse_orientation = mesh.reverse_orientation;
        transformSwapsHandedness = mesh.transformSwapsHandedness;
    }
};

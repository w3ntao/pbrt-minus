#pragma once

#include "euclidean_space/transform.h"

class TriangleMesh {
  public:
    const int triangle_num = -1;
    const int points_num = -1;

    const int *vertex_indices = nullptr;
    const Point3f *p = nullptr;
    const Normal3f *n = nullptr;
    const Vector3f *s = nullptr;
    const Point2f *uv = nullptr;
    const int *faceIndices = nullptr;

    bool reverseOrientation = false;
    bool transformSwapsHandedness = false;

    PBRT_CPU_GPU TriangleMesh(const Transform &transform, const int *_vertex_indices,
                              int _indices_num, const Point3f *_points, int _points_num)
        : triangle_num(_indices_num / 3), points_num(_points_num) {
        int *temp_vertex_indices = new int[_indices_num];
        memcpy(temp_vertex_indices, _vertex_indices, sizeof(int) * _indices_num);
        vertex_indices = temp_vertex_indices;

        auto temp_points = new Point3f[_points_num];
        for (int i = 0; i < _points_num; i++) {
            temp_points[i] = transform(_points[i]);
        }

        p = temp_points;
    }

    PBRT_CPU_GPU ~TriangleMesh() {
        delete[] vertex_indices;
        delete[] p;
        delete[] n;
        delete[] s;
        delete[] uv;
        delete[] faceIndices;
    }
};

#pragma once

#include "pbrt/euclidean_space/transform.h"

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

    bool reverse_orientation = false;
    bool transformSwapsHandedness = false;

    PBRT_GPU TriangleMesh(const Transform &render_from_object, bool _reverse_orientation,
                          const int *_vertex_indices, int num_indices, const Point3f *_points,
                          int num_points)
        : reverse_orientation(_reverse_orientation), triangle_num(num_indices / 3),
          points_num(num_points) {
        int *temp_vertex_indices = new int[num_indices];
        // TODO: new this data from cudaMallocManaged
        memcpy(temp_vertex_indices, _vertex_indices, sizeof(int) * num_indices);
        vertex_indices = temp_vertex_indices;

        auto temp_points = new Point3f[num_points];

        if (render_from_object.is_identity()) {
            memcpy(temp_points, _points, sizeof(Point3f) * num_points);
        } else {
            for (int i = 0; i < num_points; i++) {
                temp_points[i] = render_from_object(_points[i]);
            }
        }

        p = temp_points;
    }

    PBRT_GPU ~TriangleMesh() {
        delete[] vertex_indices;
        delete[] p;
        delete[] n;
        delete[] s;
        delete[] uv;
        delete[] faceIndices;
    }
};

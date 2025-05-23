#pragma once

#include <pbrt/euclidean_space/transform.h>
#include <vector>

class GPUMemoryAllocator;
class Shape;

struct TriangleMesh {
    uint triangles_num = 0;

    const int *vertex_indices = nullptr;
    const Point3f *p = nullptr;
    const Normal3f *n = nullptr;
    const Vector3f *s = nullptr;
    const Point2f *uv = nullptr;
    const int *faceIndices = nullptr;

    bool reverse_orientation = false;
    bool transformSwapsHandedness = false;

    TriangleMesh(bool _reverse_orientation, const int *_vertex_indices, uint num_indices,
                 const Point3f *_p, const Normal3f *_n, const Point2f *_uv)
        : reverse_orientation(_reverse_orientation), triangles_num(num_indices / 3),
          vertex_indices(_vertex_indices), p(_p), uv(_uv), n(_n)

    {
        reverse_orientation = _reverse_orientation;

        triangles_num = num_indices / 3;
        vertex_indices = _vertex_indices;
        p = _p;
        uv = _uv;
    }

    static std::pair<const Shape *, uint>
    build_triangles(const Transform &render_from_object, bool reverse_orientation,
                    const std::vector<Point3f> &points, const std::vector<int> &indices,
                    const std::vector<Normal3f> &normals, const std::vector<Point2f> &uv,
                    GPUMemoryAllocator &allocator);
};

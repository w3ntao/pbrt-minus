#pragma once

#include <pbrt/euclidean_space/transform.h>
#include <vector>

class GPUMemoryAllocator;
class Shape;

struct TriangleMesh {
    const int triangles_num = 0;

    const int *vertex_indices = nullptr;
    const Point3f *p = nullptr;
    const Normal3f *n = nullptr;
    const Vector3f *s = nullptr;
    const Point2f *uv = nullptr;
    const int *faceIndices = nullptr;

    const bool reverse_orientation = false;
    bool transform_swaps_handedness = false;

    TriangleMesh(const Transform &render_from_object, bool _reverse_orientation,
                 const std::vector<int> &_indices, const std::vector<Point3f> &_p,
                 const std::vector<Normal3f> &_n, const std::vector<Point2f> &_uv,
                 GPUMemoryAllocator &allocator);

    static std::pair<const Shape *, int>
    build_triangles(const Transform &render_from_object, bool reverse_orientation,
                    const std::vector<int> &indices, const std::vector<Point3f> &points,
                    const std::vector<Normal3f> &normals, const std::vector<Point2f> &uv,
                    GPUMemoryAllocator &allocator);
};

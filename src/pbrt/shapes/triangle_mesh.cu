#include "pbrt/shapes/triangle_mesh.h"

#include "pbrt/base/shape.h"
#include "pbrt/shapes/triangle.h"

template <typename T>
static __global__ void apply_transform(T *data, const Transform transform, uint length) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= length) {
        return;
    }

    data[idx] = transform(data[idx]);
}

static __global__ void init_triangles_from_mesh(Triangle *triangles, const TriangleMesh *mesh) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= mesh->triangles_num) {
        return;
    }

    triangles[worker_idx].init(worker_idx, mesh);
}

template <typename TypeOfShape>
static __global__ void init_shapes(Shape *shapes, const TypeOfShape *concrete_shapes, uint num) {
    const uint worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num) {
        return;
    }

    shapes[worker_idx].init(&concrete_shapes[worker_idx]);
}

std::pair<const Shape *, uint>
TriangleMesh::build_triangles(const Transform &render_from_object, bool reverse_orientation,
                              const std::vector<Point3f> &points, const std::vector<int> &indices,
                              const std::vector<Point2f> &uv,
                              std::vector<void *> &gpu_dynamic_pointers) {
    Point3f *gpu_points;
    CHECK_CUDA_ERROR(cudaMallocManaged(&gpu_points, sizeof(Point3f) * points.size()));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_points, points.data(), sizeof(Point3f) * points.size(),
                                cudaMemcpyHostToDevice));
    const uint threads = 1024;
    {
        const uint blocks = divide_and_ceil<uint>(points.size(), threads);
        apply_transform<<<blocks, threads>>>(gpu_points, render_from_object, points.size());
    }
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    int *gpu_indices;
    CHECK_CUDA_ERROR(cudaMallocManaged(&gpu_indices, sizeof(int) * indices.size()));
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_indices, indices.data(), sizeof(int) * indices.size(),
                                cudaMemcpyHostToDevice));

    Point2f *gpu_uv = nullptr;
    if (!uv.empty()) {
        CHECK_CUDA_ERROR(cudaMallocManaged(&gpu_uv, sizeof(Point2f) * uv.size()));
        CHECK_CUDA_ERROR(
            cudaMemcpy(gpu_uv, uv.data(), sizeof(Point2f) * uv.size(), cudaMemcpyHostToDevice));
        gpu_dynamic_pointers.push_back(gpu_uv);
    }

    TriangleMesh *mesh;
    CHECK_CUDA_ERROR(cudaMallocManaged(&mesh, sizeof(TriangleMesh)));
    mesh->init(reverse_orientation, gpu_indices, indices.size(), gpu_points, gpu_uv);

    uint num_triangles = mesh->triangles_num;
    Triangle *triangles;
    CHECK_CUDA_ERROR(cudaMallocManaged(&triangles, sizeof(Triangle) * num_triangles));
    Shape *shapes;
    CHECK_CUDA_ERROR(cudaMallocManaged(&shapes, sizeof(Shape) * num_triangles));

    {
        const uint blocks = divide_and_ceil(num_triangles, threads);
        init_triangles_from_mesh<<<blocks, threads>>>(triangles, mesh);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        init_shapes<<<blocks, threads>>>(shapes, triangles, num_triangles);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    for (auto ptr : std::vector<void *>({
             gpu_indices,
             gpu_points,
             mesh,
             triangles,
             shapes,
         })) {
        gpu_dynamic_pointers.push_back(ptr);
    }

    return {shapes, num_triangles};
}
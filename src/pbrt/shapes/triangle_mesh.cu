#include <pbrt/base/shape.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/shapes/triangle.h>
#include <pbrt/shapes/triangle_mesh.h>

template <typename T>
static __global__ void gpu_transform(T *data, const Transform transform, bool reverse_orientation,
                                     uint length) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= length) {
        return;
    }

    data[idx] = transform(data[idx]);
    if (reverse_orientation) {
        data[idx] = -data[idx];
    }
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
TriangleMesh::build_triangles(const Transform &render_from_object, const bool reverse_orientation,
                              const std::vector<Point3f> &points, const std::vector<int> &indices,
                              const std::vector<Normal3f> &normals, const std::vector<Point2f> &uv,
                              GPUMemoryAllocator &allocator) {
    auto gpu_points = allocator.allocate<Point3f>(points.size());
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_points, points.data(), sizeof(Point3f) * points.size(),
                                cudaMemcpyHostToDevice));

    constexpr uint threads = 1024;
    {
        const uint blocks = divide_and_ceil<uint>(points.size(), threads);
        if (!render_from_object.is_identity()) {
            gpu_transform<<<blocks, threads>>>(gpu_points, render_from_object, false,
                                               points.size());
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    auto gpu_indices = allocator.allocate<int>(indices.size());
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_indices, indices.data(), sizeof(int) * indices.size(),
                                cudaMemcpyHostToDevice));

    Normal3f *gpu_normals = nullptr;
    if (!normals.empty()) {
        gpu_normals = allocator.allocate<Normal3f>(normals.size());
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_normals, normals.data(), sizeof(Normal3f) * normals.size(),
                                    cudaMemcpyHostToDevice));

        if (!render_from_object.is_identity() || reverse_orientation) {
            const uint blocks = divide_and_ceil<uint>(normals.size(), threads);
            gpu_transform<<<blocks, threads>>>(gpu_normals, render_from_object, reverse_orientation,
                                               normals.size());
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
    }

    Point2f *gpu_uv = nullptr;
    if (!uv.empty()) {
        gpu_uv = allocator.allocate<Point2f>(uv.size());
        CHECK_CUDA_ERROR(
            cudaMemcpy(gpu_uv, uv.data(), sizeof(Point2f) * uv.size(), cudaMemcpyHostToDevice));
    }

    auto mesh = allocator.allocate<TriangleMesh>();
    mesh->init(reverse_orientation, gpu_indices, indices.size(), gpu_points, gpu_normals, gpu_uv);

    uint num_triangles = mesh->triangles_num;

    auto triangles = allocator.allocate<Triangle>(num_triangles);

    auto shapes = allocator.allocate<Shape>(num_triangles);

    {
        const uint blocks = divide_and_ceil(num_triangles, threads);
        init_triangles_from_mesh<<<blocks, threads>>>(triangles, mesh);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        init_shapes<<<blocks, threads>>>(shapes, triangles, num_triangles);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    return {shapes, num_triangles};
}
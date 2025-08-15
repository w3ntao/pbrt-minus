#include <pbrt/base/shape.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/shapes/triangle.h>
#include <pbrt/shapes/triangle_mesh.h>

template <typename T>
static __global__ void gpu_transform(T *data, const Transform transform, bool reverse_orientation,
                                     int length) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= length) {
        return;
    }

    data[idx] = transform(data[idx]);
    if (reverse_orientation) {
        data[idx] = -data[idx];
    }
}

static __global__ void init_triangles_from_mesh(Triangle *triangles, const TriangleMesh *mesh) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= mesh->triangles_num) {
        return;
    }

    triangles[worker_idx].init(worker_idx, mesh);
}

template <typename TypeOfShape>
static __global__ void init_shapes(Shape *shapes, const TypeOfShape *concrete_shapes, int num) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num) {
        return;
    }

    shapes[worker_idx].init(&concrete_shapes[worker_idx]);
}

std::pair<const Shape *, int>
TriangleMesh::build_triangles(const Transform &render_from_object, const bool reverse_orientation,
                              const std::vector<Point3f> &points, const std::vector<int> &indices,
                              const std::vector<Normal3f> &normals, const std::vector<Point2f> &uv,
                              GPUMemoryAllocator &allocator) {
    auto gpu_points = allocator.allocate<Point3f>(points.size());
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_points, points.data(), sizeof(Point3f) * points.size(),
                                cudaMemcpyHostToDevice));

    {
        const int blocks = divide_and_ceil<int>(points.size(), MAX_THREADS_PER_BLOCKS);
        if (!render_from_object.is_identity()) {
            gpu_transform<<<blocks, MAX_THREADS_PER_BLOCKS>>>(gpu_points, render_from_object, false,
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
            const int blocks = divide_and_ceil<int>(normals.size(), MAX_THREADS_PER_BLOCKS);
            gpu_transform<<<blocks, MAX_THREADS_PER_BLOCKS>>>(gpu_normals, render_from_object,
                                                              reverse_orientation, normals.size());
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
    *mesh = TriangleMesh(reverse_orientation, gpu_indices, indices.size(), gpu_points, gpu_normals,
                         gpu_uv);

    int num_triangles = mesh->triangles_num;

    auto triangles = allocator.allocate<Triangle>(num_triangles);

    auto shapes = allocator.allocate<Shape>(num_triangles);

    {
        const int blocks = divide_and_ceil(num_triangles, MAX_THREADS_PER_BLOCKS);
        init_triangles_from_mesh<<<blocks, MAX_THREADS_PER_BLOCKS>>>(triangles, mesh);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        init_shapes<<<blocks, MAX_THREADS_PER_BLOCKS>>>(shapes, triangles, num_triangles);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }

    return {shapes, num_triangles};
}
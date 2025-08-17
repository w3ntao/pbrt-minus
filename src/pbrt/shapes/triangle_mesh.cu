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

    triangles[worker_idx] = Triangle(worker_idx, mesh);
}

template <typename TypeOfShape>
static __global__ void init_shapes(Shape *shapes, const TypeOfShape *concrete_shapes, int num) {
    const int worker_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (worker_idx >= num) {
        return;
    }

    shapes[worker_idx] = Shape(&concrete_shapes[worker_idx]);
}

TriangleMesh::TriangleMesh(const Transform &render_from_object, const bool _reverse_orientation,
                           const std::vector<int> &_indices, const std::vector<Point3f> &_p,
                           const std::vector<Normal3f> &_n, const std::vector<Point2f> &_uv,
                           GPUMemoryAllocator &allocator)
    : triangles_num(_indices.size() / 3), reverse_orientation(_reverse_orientation),
      transform_swaps_handedness(render_from_object.swaps_handedness()) {

    auto gpu_points = allocator.allocate<Point3f>(_p.size());
    CHECK_CUDA_ERROR(
        cudaMemcpy(gpu_points, _p.data(), sizeof(Point3f) * _p.size(), cudaMemcpyHostToDevice));
    {
        const int blocks = divide_and_ceil<int>(_p.size(), MAX_THREADS_PER_BLOCKS);
        if (!render_from_object.is_identity()) {
            gpu_transform<<<blocks, MAX_THREADS_PER_BLOCKS>>>(gpu_points, render_from_object, false,
                                                              _p.size());
        }
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
    p = gpu_points;

    auto gpu_indices = allocator.allocate<int>(_indices.size());
    CHECK_CUDA_ERROR(cudaMemcpy(gpu_indices, _indices.data(), sizeof(int) * _indices.size(),
                                cudaMemcpyHostToDevice));
    vertex_indices = gpu_indices;

    if (!_n.empty()) {
        auto gpu_normals = allocator.allocate<Normal3f>(_n.size());
        CHECK_CUDA_ERROR(cudaMemcpy(gpu_normals, _n.data(), sizeof(Normal3f) * _n.size(),
                                    cudaMemcpyHostToDevice));

        if (!render_from_object.is_identity() || reverse_orientation) {
            const int blocks = divide_and_ceil<int>(_n.size(), MAX_THREADS_PER_BLOCKS);
            gpu_transform<<<blocks, MAX_THREADS_PER_BLOCKS>>>(gpu_normals, render_from_object,
                                                              reverse_orientation, _n.size());
            CHECK_CUDA_ERROR(cudaGetLastError());
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        }
        n = gpu_normals;
    }

    if (!_uv.empty()) {
        auto gpu_uv = allocator.allocate<Point2f>(_uv.size());
        CHECK_CUDA_ERROR(
            cudaMemcpy(gpu_uv, _uv.data(), sizeof(Point2f) * _uv.size(), cudaMemcpyHostToDevice));
        uv = gpu_uv;
    }
}

std::pair<const Shape *, int>
TriangleMesh::build_triangles(const Transform &render_from_object, const bool reverse_orientation,
                              const std::vector<int> &indices, const std::vector<Point3f> &points,
                              const std::vector<Normal3f> &normals, const std::vector<Point2f> &uv,
                              GPUMemoryAllocator &allocator) {
    const auto mesh = allocator.create<TriangleMesh>(render_from_object, reverse_orientation,
                                                     indices, points, normals, uv, allocator);
    const int num_triangles = mesh->triangles_num;

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

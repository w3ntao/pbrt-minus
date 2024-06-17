#include "pbrt/base/primitive.h"

#include "pbrt/base/light.h"

#include "pbrt/primitives/simple_primitives.h"
#include "pbrt/primitives/geometric_primitive.h"

static __global__ void init_simple_primitives(SimplePrimitive *simple_primitives,
                                              const Shape *shapes, const Material *material,
                                              uint num) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num) {
        return;
    }

    simple_primitives[idx].init(&shapes[idx], material);
}

static __global__ void init_geometric_primitives(GeometricPrimitive *geometric_primitives,
                                                 const Shape *shapes, const Material *material,
                                                 const Light *diffuse_area_lights, uint num) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num) {
        return;
    }

    geometric_primitives[idx].init(&shapes[idx], material, &diffuse_area_lights[idx]);
}

template <typename TypeOfPrimitive>
static __global__ void init_primitives(Primitive *primitives, TypeOfPrimitive *_primitives,
                                       uint num) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num) {
        return;
    }

    primitives[idx].init(&_primitives[idx]);
}

const Primitive *Primitive::create_simple_primitives(const Shape *shapes, const Material *material,
                                                     uint num,
                                                     std::vector<void *> &gpu_dynamic_pointers) {
    const uint threads = 1024;
    const uint blocks = divide_and_ceil(num, threads);

    SimplePrimitive *simple_primitives;
    CHECK_CUDA_ERROR(cudaMallocManaged(&simple_primitives, sizeof(SimplePrimitive) * num));
    Primitive *primitives;
    CHECK_CUDA_ERROR(cudaMallocManaged(&primitives, sizeof(Primitive) * num));

    init_simple_primitives<<<blocks, threads>>>(simple_primitives, shapes, material, num);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    init_primitives<<<blocks, threads>>>(primitives, simple_primitives, num);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    gpu_dynamic_pointers.push_back(simple_primitives);
    gpu_dynamic_pointers.push_back(primitives);

    return primitives;
}

const Primitive *Primitive::create_geometric_primitives(const Shape *shapes,
                                                        const Material *material,
                                                        const Light *diffuse_area_light, uint num,
                                                        std::vector<void *> &gpu_dynamic_pointers) {
    const uint threads = 1024;
    const uint blocks = divide_and_ceil(num, threads);

    GeometricPrimitive *geometric_primitives;
    CHECK_CUDA_ERROR(cudaMallocManaged(&geometric_primitives, sizeof(GeometricPrimitive) * num));

    Primitive *primitives;
    CHECK_CUDA_ERROR(cudaMallocManaged(&primitives, sizeof(Primitive) * num));

    gpu_dynamic_pointers.push_back(geometric_primitives);
    gpu_dynamic_pointers.push_back(primitives);

    init_geometric_primitives<<<blocks, threads>>>(geometric_primitives, shapes, material,
                                                   diffuse_area_light, num);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    init_primitives<<<blocks, threads>>>(primitives, geometric_primitives, num);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return primitives;
}

PBRT_CPU_GPU
void Primitive::init(const SimplePrimitive *simple_primitive) {
    type = Type::simple_primitive;
    ptr = simple_primitive;
}

PBRT_CPU_GPU
void Primitive::init(const GeometricPrimitive *geometric_primitive) {
    type = Type::geometric_primitive;
    ptr = geometric_primitive;
}

PBRT_CPU_GPU
Bounds3f Primitive::bounds() const {
    switch (type) {
    case (Type::simple_primitive): {
        return ((SimplePrimitive *)ptr)->bounds();
    }

    case (Type::geometric_primitive): {
        return ((GeometricPrimitive *)ptr)->bounds();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_GPU
bool Primitive::fast_intersect(const Ray &ray, FloatType t_max) const {
    switch (type) {
    case (Type::simple_primitive): {
        return ((SimplePrimitive *)ptr)->fast_intersect(ray, t_max);
    }

    case (Type::geometric_primitive): {
        return ((GeometricPrimitive *)ptr)->fast_intersect(ray, t_max);
    }
    }

    REPORT_FATAL_ERROR();
    return false;
}

PBRT_GPU
cuda::std::optional<ShapeIntersection> Primitive::intersect(const Ray &ray, FloatType t_max) const {
    switch (type) {
    case (Type::simple_primitive): {
        return ((SimplePrimitive *)ptr)->intersect(ray, t_max);
    }

    case (Type::geometric_primitive): {
        return ((GeometricPrimitive *)ptr)->intersect(ray, t_max);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

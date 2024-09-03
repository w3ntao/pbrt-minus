#include "pbrt/base/light.h"
#include "pbrt/base/primitive.h"
#include "pbrt/primitives/geometric_primitive.h"
#include "pbrt/primitives/simple_primitive.h"
#include "pbrt/primitives/transformed_primitive.h"

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

static __global__ void init_transformed_primitives(Primitive *primitives,
                                                   TransformedPrimitive *transformed_primitives,
                                                   const Primitive *base_primitives,
                                                   const Transform render_from_primitive,
                                                   uint num) {
    uint idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num) {
        return;
    }

    transformed_primitives[idx].init(&base_primitives[idx], render_from_primitive);
    primitives[idx].init(&transformed_primitives[idx]);
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

const Primitive *
Primitive::create_transformed_primitives(const Primitive *base_primitives,
                                         const Transform render_from_primitive, uint num,
                                         std::vector<void *> &gpu_dynamic_pointers) {
    const uint threads = 1024;
    const uint blocks = divide_and_ceil(num, threads);

    TransformedPrimitive *transformed_primitives;
    CHECK_CUDA_ERROR(
        cudaMallocManaged(&transformed_primitives, sizeof(TransformedPrimitive) * num));
    Primitive *primitives;
    CHECK_CUDA_ERROR(cudaMallocManaged(&primitives, sizeof(Primitive) * num));

    gpu_dynamic_pointers.push_back(transformed_primitives);
    gpu_dynamic_pointers.push_back(primitives);

    init_transformed_primitives<<<blocks, threads>>>(primitives, transformed_primitives,
                                                     base_primitives, render_from_primitive, num);

    return primitives;
}

PBRT_CPU_GPU
void Primitive::init(const GeometricPrimitive *geometric_primitive) {
    type = Type::geometric;
    ptr = geometric_primitive;
}

PBRT_CPU_GPU
void Primitive::init(const SimplePrimitive *simple_primitive) {
    type = Type::simple;
    ptr = simple_primitive;
}

PBRT_CPU_GPU
void Primitive::init(const TransformedPrimitive *transformed_primitive) {
    type = Type::transformed;
    ptr = transformed_primitive;
}

PBRT_CPU_GPU
const Material *Primitive::get_material() const {
    switch (type) {
    case (Type::geometric): {
        return ((GeometricPrimitive *)ptr)->get_material();
    }

    case (Type::simple): {
        return ((SimplePrimitive *)ptr)->get_material();
    }

    case (Type::transformed): {
        return ((TransformedPrimitive *)ptr)->get_material();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
Bounds3f Primitive::bounds() const {
    switch (type) {
    case (Type::geometric): {
        return ((GeometricPrimitive *)ptr)->bounds();
    }

    case (Type::simple): {
        return ((SimplePrimitive *)ptr)->bounds();
    }

    case (Type::transformed): {
        return ((TransformedPrimitive *)ptr)->bounds();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_GPU
bool Primitive::fast_intersect(const Ray &ray, FloatType t_max) const {
    switch (type) {
    case (Type::geometric): {
        return ((GeometricPrimitive *)ptr)->fast_intersect(ray, t_max);
    }

    case (Type::simple): {
        return ((SimplePrimitive *)ptr)->fast_intersect(ray, t_max);
    }

    case (Type::transformed): {
        return ((TransformedPrimitive *)ptr)->fast_intersect(ray, t_max);
    }
    }

    REPORT_FATAL_ERROR();
    return false;
}

PBRT_GPU
cuda::std::optional<ShapeIntersection> Primitive::intersect(const Ray &ray, FloatType t_max) const {
    switch (type) {

    case (Type::geometric): {
        return ((GeometricPrimitive *)ptr)->intersect(ray, t_max);
    }

    case (Type::simple): {
        return ((SimplePrimitive *)ptr)->intersect(ray, t_max);
    }

    case (Type::transformed): {
        return ((TransformedPrimitive *)ptr)->intersect(ray, t_max);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

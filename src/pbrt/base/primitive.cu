#include <pbrt/accelerator/hlbvh.h>
#include <pbrt/base/light.h>
#include <pbrt/base/material.h>
#include <pbrt/base/primitive.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/primitives/geometric_primitive.h>
#include <pbrt/primitives/simple_primitive.h>
#include <pbrt/primitives/transformed_primitive.h>

static __global__ void init_simple_primitives(SimplePrimitive *simple_primitives,
                                              const Shape *shapes, const Material *material,
                                              const int num) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num) {
        return;
    }

    simple_primitives[idx] = SimplePrimitive(&shapes[idx], material);
}

static __global__ void init_geometric_primitives(GeometricPrimitive *geometric_primitives,
                                                 const Shape *shapes, const Material *material,
                                                 const Light *diffuse_area_lights,
                                                 const MediumInterface *medium_interface,
                                                 const int num) {
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num) {
        return;
    }

    geometric_primitives[idx] = GeometricPrimitive(
        &shapes[idx], material, diffuse_area_lights ? &diffuse_area_lights[idx] : nullptr,
        medium_interface);
}

static __global__ void init_transformed_primitives(Primitive *primitives,
                                                   TransformedPrimitive *transformed_primitives,
                                                   const Primitive *base_primitives,
                                                   const Transform render_from_primitive, int num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num) {
        return;
    }

    transformed_primitives[idx] =
        TransformedPrimitive(&base_primitives[idx], render_from_primitive);
    primitives[idx] = Primitive(&transformed_primitives[idx]);
}

template <typename TypeOfPrimitive>
static __global__ void init_primitives(Primitive *primitives, TypeOfPrimitive *_primitives,
                                       int num) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num) {
        return;
    }

    primitives[idx] = Primitive(&_primitives[idx]);
}

const Primitive *Primitive::create_geometric_primitives(const Shape *shapes,
                                                        const Material *material,
                                                        const Light *diffuse_area_light,
                                                        const MediumInterface *medium_interface,
                                                        int num, GPUMemoryAllocator &allocator) {
    const int blocks = divide_and_ceil(num, MAX_THREADS_PER_BLOCKS);

    auto geometric_primitives = allocator.allocate<GeometricPrimitive>(num);

    auto primitives = allocator.allocate<Primitive>(num);

    init_geometric_primitives<<<blocks, MAX_THREADS_PER_BLOCKS>>>(
        geometric_primitives, shapes, material, diffuse_area_light, medium_interface, num);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    init_primitives<<<blocks, MAX_THREADS_PER_BLOCKS>>>(primitives, geometric_primitives, num);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return primitives;
}

const Primitive *Primitive::create_simple_primitives(const Shape *shapes, const Material *material,
                                                     int num, GPUMemoryAllocator &allocator) {
    const int blocks = divide_and_ceil(num, MAX_THREADS_PER_BLOCKS);

    auto simple_primitives = allocator.allocate<SimplePrimitive>(num);

    auto primitives = allocator.allocate<Primitive>(num);

    init_simple_primitives<<<blocks, MAX_THREADS_PER_BLOCKS>>>(simple_primitives, shapes, material,
                                                               num);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    init_primitives<<<blocks, MAX_THREADS_PER_BLOCKS>>>(primitives, simple_primitives, num);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    return primitives;
}

const Primitive *Primitive::create_transformed_primitives(const Primitive *base_primitives,
                                                          const Transform &render_from_primitive,
                                                          int num, GPUMemoryAllocator &allocator) {
    const int blocks = divide_and_ceil(num, MAX_THREADS_PER_BLOCKS);

    auto transformed_primitives = allocator.allocate<TransformedPrimitive>(num);

    auto primitives = allocator.allocate<Primitive>(num);

    init_transformed_primitives<<<blocks, MAX_THREADS_PER_BLOCKS>>>(
        primitives, transformed_primitives, base_primitives, render_from_primitive, num);

    return primitives;
}

PBRT_CPU_GPU
const Material *Primitive::get_material() const {
    switch (type) {
    case Type::bvh: {
        REPORT_FATAL_ERROR();
        return nullptr;
    }

    case Type::geometric: {
        return static_cast<const GeometricPrimitive *>(ptr)->get_material();
    }

    case Type::simple: {
        return static_cast<const SimplePrimitive *>(ptr)->get_material();
    }

    case Type::transformed: {
        return static_cast<const TransformedPrimitive *>(ptr)->get_material();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
Bounds3f Primitive::bounds() const {
    switch (type) {
    case Type::bvh: {
        return static_cast<const HLBVH *>(ptr)->bounds();
    }

    case Type::geometric: {
        return static_cast<const GeometricPrimitive *>(ptr)->bounds();
    }

    case Type::simple: {
        return static_cast<const SimplePrimitive *>(ptr)->bounds();
    }

    case Type::transformed: {
        return static_cast<const TransformedPrimitive *>(ptr)->bounds();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

void Primitive::record_material(std::map<std::string, int> &counter) const {
    const auto count_material = [](const Material *material, std::map<std::string, int> &counter) {
        const auto name = material
                              ? Material::material_type_to_string(material->get_material_type())
                              : "interface";

        if (!counter.contains(name)) {
            counter[name] = 1;
            return;
        }

        counter[name] += 1;
    };

    if (type == Type::geometric) {
        const auto material = static_cast<const GeometricPrimitive *>(ptr)->get_material();
        count_material(material, counter);

        return;
    }

    if (type == Type::simple) {
        const auto material = static_cast<const SimplePrimitive *>(ptr)->get_material();
        count_material(material, counter);

        return;
    }

    if (type == Type::transformed) {
        const auto primitive = static_cast<const TransformedPrimitive *>(ptr)->get_primitive();
        primitive->record_material(counter);

        return;
    }

    if (type == Type::bvh) {
        const auto [primitives, num] = static_cast<const HLBVH *>(ptr)->get_primitives();

        for (int idx = 0; idx < num; idx++) {
            primitives[idx]->record_material(counter);
        }

        return;
    }

    REPORT_FATAL_ERROR();
}

PBRT_CPU_GPU
bool Primitive::fast_intersect(const Ray &ray, const Real t_max) const {
    switch (type) {
    case Type::bvh: {
        return static_cast<const HLBVH *>(ptr)->fast_intersect(ray, t_max);
    }

    case Type::geometric: {
        return static_cast<const GeometricPrimitive *>(ptr)->fast_intersect(ray, t_max);
    }

    case Type::simple: {
        return static_cast<const SimplePrimitive *>(ptr)->fast_intersect(ray, t_max);
    }

    case Type::transformed: {
        return static_cast<const TransformedPrimitive *>(ptr)->fast_intersect(ray, t_max);
    }
    }

    REPORT_FATAL_ERROR();
    return false;
}

PBRT_CPU_GPU
pbrt::optional<ShapeIntersection> Primitive::intersect(const Ray &ray, const Real t_max) const {
    switch (type) {
    case Type::bvh: {
        return static_cast<const HLBVH *>(ptr)->intersect(ray, t_max);
    }

    case Type::geometric: {
        return static_cast<const GeometricPrimitive *>(ptr)->intersect(ray, t_max);
    }

    case Type::simple: {
        return static_cast<const SimplePrimitive *>(ptr)->intersect(ray, t_max);
    }

    case Type::transformed: {
        return static_cast<const TransformedPrimitive *>(ptr)->intersect(ray, t_max);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

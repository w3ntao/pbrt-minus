#pragma once

#include <map>
#include <pbrt/base/interaction.h>
#include <pbrt/euclidean_space/bounds3.h>
#include <pbrt/euclidean_space/transform.h>
#include <pbrt/gpu/macro.h>

class GPUMemoryAllocator;
class HLBVH;
class Shape;
class Material;

class GeometricPrimitive;
class SimplePrimitive;
class TransformedPrimitive;

class Primitive {
  public:
    enum class Type {
        bvh,
        geometric,
        simple,
        transformed,
    };

    PBRT_CPU_GPU
    explicit Primitive(const HLBVH *hlbvh) : type(Type::bvh), ptr(hlbvh) {}

    PBRT_CPU_GPU
    explicit Primitive(const GeometricPrimitive *geometric_primitive)
        : type(Type::geometric), ptr(geometric_primitive) {}

    PBRT_CPU_GPU
    explicit Primitive(const SimplePrimitive *simple_primitive)
        : type(Type::simple), ptr(simple_primitive) {}

    PBRT_CPU_GPU
    explicit Primitive(const TransformedPrimitive *transformed_primitive)
        : type(Type::transformed), ptr(transformed_primitive) {}

    static const Primitive *create_geometric_primitives(const Shape *shapes,
                                                        const Material *material,
                                                        const Light *diffuse_area_light,
                                                        const MediumInterface *medium_interface,
                                                        int num, GPUMemoryAllocator &allocator);

    static const Primitive *create_simple_primitives(const Shape *shapes, const Material *material,
                                                     int num, GPUMemoryAllocator &allocator);

    static const Primitive *create_transformed_primitives(const Primitive *base_primitives,
                                                          const Transform &render_from_primitive,
                                                          int num, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    const Material *get_material() const;

    PBRT_CPU_GPU
    Bounds3f bounds() const;

    void record_material(std::map<std::string, int> &counter) const;

    PBRT_CPU_GPU
    bool fast_intersect(const Ray &ray, Real t_max) const;

    PBRT_CPU_GPU
    pbrt::optional<ShapeIntersection> intersect(const Ray &ray, Real t_max) const;

  private:
    Type type;
    const void *ptr = nullptr;
};

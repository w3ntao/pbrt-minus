#pragma once

#include <vector>
#include <cuda/std/optional>

#include "pbrt/base/interaction.h"
#include "pbrt/euclidean_space/bounds3.h"
#include "pbrt/euclidean_space/transform.h"
#include "pbrt/util/macro.h"

class Shape;
class Material;

class GeometricPrimitive;
class SimplePrimitive;
class TransformedPrimitive;

class Primitive {
  public:
    enum class Type {
        geometric,
        simple,
        transformed,
    };

    static const Primitive *create_geometric_primitives(const Shape *shapes,
                                                        const Material *material,
                                                        const Light *diffuse_area_light, uint num,
                                                        std::vector<void *> &gpu_dynamic_pointers);

    static const Primitive *create_simple_primitives(const Shape *shapes, const Material *material,
                                                     uint num,
                                                     std::vector<void *> &gpu_dynamic_pointers);

    static const Primitive *
    create_transformed_primitives(const Primitive *base_primitives,
                                  const Transform render_from_primitive, uint num,
                                  std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU void init(const GeometricPrimitive *geometric_primitive);

    PBRT_CPU_GPU
    void init(const SimplePrimitive *simple_primitive);

    PBRT_CPU_GPU
    void init(const TransformedPrimitive *transformed_primitive);

    PBRT_CPU_GPU
    Bounds3f bounds() const;

    PBRT_GPU
    bool fast_intersect(const Ray &ray, FloatType t_max) const;

    PBRT_GPU
    cuda::std::optional<ShapeIntersection> intersect(const Ray &ray, FloatType t_max) const;

  private:
    Type type;
    const void *ptr;
};

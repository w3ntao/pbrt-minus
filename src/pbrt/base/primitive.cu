#include "pbrt/base/primitive.h"
#include "pbrt/primitives/simple_primitives.h"
#include "pbrt/primitives/geometric_primitive.h"

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

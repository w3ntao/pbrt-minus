#include "pbrt/base/primitive.h"
#include "pbrt/primitives/simple_primitives.h"
#include "pbrt/primitives/geometric_primitive.h"

PBRT_CPU_GPU
void Primitive::init(SimplePrimitive *simple_primitive) {
    primitive_type = PrimitiveType::simple_primitive;
    primitive_ptr = simple_primitive;
}

PBRT_CPU_GPU
void Primitive::init(GeometricPrimitive *geometric_primitive) {
    primitive_type = PrimitiveType::geometric_primitive;
    primitive_ptr = geometric_primitive;
}

PBRT_CPU_GPU
Bounds3f Primitive::bounds() const {
    switch (primitive_type) {
    case (PrimitiveType::simple_primitive): {
        return ((SimplePrimitive *)primitive_ptr)->bounds();
    }

    case (PrimitiveType::geometric_primitive): {
        return ((GeometricPrimitive *)primitive_ptr)->bounds();
    }
    }

    report_function_error_and_exit(__func__);
    return {};
}

PBRT_GPU
bool Primitive::fast_intersect(const Ray &ray, FloatType t_max) const {
    switch (primitive_type) {
    case (PrimitiveType::simple_primitive): {
        return ((SimplePrimitive *)primitive_ptr)->fast_intersect(ray, t_max);
    }

    case (PrimitiveType::geometric_primitive): {
        return ((GeometricPrimitive *)primitive_ptr)->fast_intersect(ray, t_max);
    }
    }

    report_function_error_and_exit(__func__);
    return false;
}

PBRT_GPU
std::optional<ShapeIntersection> Primitive::intersect(const Ray &ray, FloatType t_max) const {
    switch (primitive_type) {
    case (PrimitiveType::simple_primitive): {
        return ((SimplePrimitive *)primitive_ptr)->intersect(ray, t_max);
    }

    case (PrimitiveType::geometric_primitive): {
        return ((GeometricPrimitive *)primitive_ptr)->intersect(ray, t_max);
    }
    }

    report_function_error_and_exit(__func__);
    return {};
}

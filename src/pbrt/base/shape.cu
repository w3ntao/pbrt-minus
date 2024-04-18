#include "pbrt/base/shape.h"
#include "pbrt/shapes/triangle.h"

PBRT_CPU_GPU
void Shape::init(const Triangle *triangle) {
    shape_type = ShapeType::triangle;
    shape_ptr = (void *)triangle;
}

PBRT_CPU_GPU
Bounds3f Shape::bounds() const {
    switch (shape_type) {
    case (ShapeType::triangle): {
        return ((Triangle *)shape_ptr)->bounds();
    }
    }

    report_error();
    return {};
}

PBRT_CPU_GPU
FloatType Shape::area() const {
    switch (shape_type) {
    case (ShapeType::triangle): {
        return ((Triangle *)shape_ptr)->area();
    }
    }

    report_error();
    return NAN;
}

PBRT_GPU
bool Shape::fast_intersect(const Ray &ray, FloatType t_max) const {
    switch (shape_type) {
    case (ShapeType::triangle): {
        return ((Triangle *)shape_ptr)->fast_intersect(ray, t_max);
    }
    }

    report_error();
    return false;
}

PBRT_GPU
std::optional<ShapeIntersection> Shape::intersect(const Ray &ray, FloatType t_max) const {
    switch (shape_type) {
    case (ShapeType::triangle): {
        return ((Triangle *)shape_ptr)->intersect(ray, t_max);
    }
    }

    report_error();
    return {};
}

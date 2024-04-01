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

    printf("\nShape::bounds(): not implemented for this type\n\n");
#if defined(__CUDA_ARCH__)
    asm("trap;");
#else
    throw std::runtime_error("Shape::bounds(): not implemented for this type\n");
#endif
}

PBRT_GPU
bool Shape::fast_intersect(const Ray &ray, double t_max) const {
    switch (shape_type) {
    case (ShapeType::triangle): {
        return ((Triangle *)shape_ptr)->fast_intersect(ray, t_max);
    }
    }

    printf("\nShape::fast_intersect(): not implemented for this type\n\n");
    asm("trap;");
}

PBRT_GPU
std::optional<ShapeIntersection> Shape::intersect(const Ray &ray, double t_max) const {
    switch (shape_type) {
    case (ShapeType::triangle): {
        return ((Triangle *)shape_ptr)->intersect(ray, t_max);
    }
    }

    printf("\nShape::intersect(): not implemented for this type\n\n");
    asm("trap;");
}

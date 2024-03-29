#pragma once

#include "pbrt/shapes/triangle.h"

namespace {
enum class ShapeType { triangle };
}

class Shape {
  public:
    ShapeType type;
    void *data_ptr;

    PBRT_CPU_GPU void init(const Triangle *triangle) {
        type = ShapeType::triangle;
        data_ptr = (void *)triangle;
    }

    PBRT_CPU_GPU Bounds3f bounds() const {
        switch (type) {
        case (ShapeType::triangle): {
            return ((Triangle *)data_ptr)->bounds();
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
    bool fast_intersect(const Ray &ray, double t_max) const {
        switch (type) {
        case (ShapeType::triangle): {
            return ((Triangle *)data_ptr)->fast_intersect(ray, t_max);
        }
        }

        printf("\nShape::fast_intersect(): not implemented for this type\n\n");
        asm("trap;");
    }

    PBRT_GPU
    std::optional<ShapeIntersection> intersect(const Ray &ray, double t_max) const {
        switch (type) {
        case (ShapeType::triangle): {
            return ((Triangle *)data_ptr)->intersect(ray, t_max);
        }
        }

        printf("\nShape::intersect(): not implemented for this type\n\n");
        asm("trap;");
    }
};

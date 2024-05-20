#include "pbrt/base/shape.h"
#include "pbrt/shapes/triangle.h"

PBRT_CPU_GPU
void Shape::init(const Triangle *triangle) {
    type = Type::triangle;
    ptr = triangle;
}

PBRT_CPU_GPU
Bounds3f Shape::bounds() const {
    switch (type) {
    case (Type::triangle): {
        return ((Triangle *)ptr)->bounds();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
FloatType Shape::area() const {
    switch (type) {
    case (Type::triangle): {
        return ((Triangle *)ptr)->area();
    }
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

PBRT_GPU
bool Shape::fast_intersect(const Ray &ray, FloatType t_max) const {
    switch (type) {
    case (Type::triangle): {
        return ((Triangle *)ptr)->fast_intersect(ray, t_max);
    }
    }

    REPORT_FATAL_ERROR();
    return false;
}

PBRT_GPU
cuda::std::optional<ShapeIntersection> Shape::intersect(const Ray &ray, FloatType t_max) const {
    switch (type) {
    case (Type::triangle): {
        return ((Triangle *)ptr)->intersect(ray, t_max);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_GPU
cuda::std::optional<ShapeSample> Shape::sample(const ShapeSampleContext &ctx,
                                               const Point2f u) const {
    switch (type) {
    case (Type::triangle): {
        return ((Triangle *)ptr)->sample(ctx, u);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

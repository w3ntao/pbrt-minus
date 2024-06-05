#include "pbrt/base/shape.h"

#include "pbrt/shapes/triangle.h"
#include "pbrt/shapes/sphere.h"

const Shape *Shape::create_sphere(const Transform &render_from_object,
                                  const Transform &object_from_render, bool reverse_orientation,
                                  const ParameterDict &parameters,
                                  std::vector<void *> &gpu_dynamic_pointers) {
    auto sphere = Sphere::create(render_from_object, object_from_render, reverse_orientation,
                                 parameters, gpu_dynamic_pointers);

    Shape *shape;
    CHECK_CUDA_ERROR(cudaMallocManaged(&shape, sizeof(Shape)));
    gpu_dynamic_pointers.push_back(shape);

    shape->init(sphere);
    return shape;
}

PBRT_CPU_GPU
void Shape::init(const Triangle *triangle) {
    type = Type::triangle;
    shape = triangle;
}

PBRT_CPU_GPU
void Shape::init(const Sphere *sphere) {
    type = Type::sphere;
    shape = sphere;
}

PBRT_CPU_GPU
Bounds3f Shape::bounds() const {
    switch (type) {
    case (Type::triangle): {
        return ((Triangle *)shape)->bounds();
    }

    case (Type::sphere): {
        return ((Sphere *)shape)->bounds();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
FloatType Shape::area() const {
    switch (type) {
    case (Type::triangle): {
        return ((Triangle *)shape)->area();
    }
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

PBRT_GPU
bool Shape::fast_intersect(const Ray &ray, FloatType t_max) const {
    switch (type) {
    case (Type::triangle): {
        return ((Triangle *)shape)->fast_intersect(ray, t_max);
    }

    case (Type::sphere): {
        return ((Sphere *)shape)->fast_intersect(ray, t_max);
    }
    }

    REPORT_FATAL_ERROR();
    return false;
}

PBRT_GPU
cuda::std::optional<ShapeIntersection> Shape::intersect(const Ray &ray, FloatType t_max) const {
    switch (type) {
    case (Type::triangle): {
        return ((Triangle *)shape)->intersect(ray, t_max);
    }

    case (Type::sphere): {
        return ((Sphere *)shape)->intersect(ray, t_max);
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
        return ((Triangle *)shape)->sample(ctx, u);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

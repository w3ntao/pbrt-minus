#include "pbrt/base/shape.h"

#include "pbrt/shapes/disk.h"
#include "pbrt/shapes/sphere.h"
#include "pbrt/shapes/triangle.h"

const Shape *Shape::create_disk(const Transform &render_from_object,
                                const Transform &object_from_render, bool reverse_orientation,
                                const ParameterDictionary &parameters,
                                std::vector<void *> &gpu_dynamic_pointers) {
    auto disk = Disk::create(render_from_object, object_from_render, reverse_orientation,
                             parameters, gpu_dynamic_pointers);

    Shape *shape;
    CHECK_CUDA_ERROR(cudaMallocManaged(&shape, sizeof(Shape)));
    gpu_dynamic_pointers.push_back(shape);

    shape->init(disk);

    return shape;
}

const Shape *Shape::create_sphere(const Transform &render_from_object,
                                  const Transform &object_from_render, bool reverse_orientation,
                                  const ParameterDictionary &parameters,
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
void Shape::init(const Disk *disk) {
    type = Type::disk;
    ptr = disk;
}

PBRT_CPU_GPU
void Shape::init(const Triangle *triangle) {
    type = Type::triangle;
    ptr = triangle;
}

PBRT_CPU_GPU
void Shape::init(const Sphere *sphere) {
    type = Type::sphere;
    ptr = sphere;
}

PBRT_CPU_GPU
Bounds3f Shape::bounds() const {
    switch (type) {
    case (Type::disk): {
        return ((Disk *)ptr)->bounds();
    }

    case (Type::sphere): {
        return ((Sphere *)ptr)->bounds();
    }

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
    case (Type::disk): {
        return ((Disk *)ptr)->area();
    }

    case (Type::sphere): {
        return ((Sphere *)ptr)->area();
    }

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
    case (Type::disk): {
        return ((Disk *)ptr)->fast_intersect(ray, t_max);
    }

    case (Type::triangle): {
        return ((Triangle *)ptr)->fast_intersect(ray, t_max);
    }

    case (Type::sphere): {
        return ((Sphere *)ptr)->fast_intersect(ray, t_max);
    }
    }

    REPORT_FATAL_ERROR();
    return false;
}

PBRT_GPU
cuda::std::optional<ShapeIntersection> Shape::intersect(const Ray &ray, FloatType t_max) const {
    switch (type) {
    case (Type::disk): {
        return ((Disk *)ptr)->intersect(ray, t_max);
    }

    case (Type::triangle): {
        return ((Triangle *)ptr)->intersect(ray, t_max);
    }

    case (Type::sphere): {
        return ((Sphere *)ptr)->intersect(ray, t_max);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_GPU
cuda::std::optional<ShapeSample> Shape::sample(const ShapeSampleContext &ctx,
                                               const Point2f &u) const {
    switch (type) {
    case (Type::disk): {
        return ((Disk *)ptr)->sample(ctx, u);
    }

    case (Type::triangle): {
        return ((Triangle *)ptr)->sample(ctx, u);
    }

    case (Type::sphere): {
        return ((Sphere *)ptr)->sample(ctx, u);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

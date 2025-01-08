#include "pbrt/base/shape.h"
#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/shapes/disk.h"
#include "pbrt/shapes/loop_subdivide.h"
#include "pbrt/shapes/sphere.h"
#include "pbrt/shapes/tri_quad_mesh.h"
#include "pbrt/shapes/triangle.h"

std::pair<const Shape *, uint>
Shape::create(const std::string &type_of_shape, const Transform &render_from_object,
              const Transform &object_from_render, bool reverse_orientation,
              const ParameterDictionary &parameters, std::vector<void *> &gpu_dynamic_pointers) {
    if (type_of_shape == "disk") {
        auto disk = Disk::create(render_from_object, object_from_render, reverse_orientation,
                                 parameters, gpu_dynamic_pointers);

        Shape *shape;
        CHECK_CUDA_ERROR(cudaMallocManaged(&shape, sizeof(Shape)));
        gpu_dynamic_pointers.push_back(shape);

        shape->init(disk);
        return {shape, 1};
    }

    if (type_of_shape == "sphere") {
        auto sphere = Sphere::create(render_from_object, object_from_render, reverse_orientation,
                                     parameters, gpu_dynamic_pointers);

        Shape *shape;
        CHECK_CUDA_ERROR(cudaMallocManaged(&shape, sizeof(Shape)));
        gpu_dynamic_pointers.push_back(shape);

        shape->init(sphere);
        return {shape, 1};
    }

    if (type_of_shape == "plymesh") {
        auto file_path = parameters.root + "/" + parameters.get_one_string("filename");
        auto ply_mesh = TriQuadMesh::read_ply(file_path);

        const Shape *shapes = nullptr;
        uint num_shapes = 0;

        if (!ply_mesh.triIndices.empty()) {
            auto result = TriangleMesh::build_triangles(render_from_object, reverse_orientation,
                                                        ply_mesh.p, ply_mesh.triIndices, ply_mesh.n,
                                                        ply_mesh.uv, gpu_dynamic_pointers);
            shapes = result.first;
            num_shapes = result.second;
        }

        if (!ply_mesh.quadIndices.empty()) {
            printf("\n%s(): Shape::plymesh.quadIndices not implemented\n", __func__);
            REPORT_FATAL_ERROR();
        }

        return {shapes, num_shapes};
    }

    if (type_of_shape == "trianglemesh") {
        auto uv = parameters.get_point2_array("uv");
        auto indices = parameters.get_integers("indices");
        auto points = parameters.get_point3_array("P");
        auto normals = parameters.get_normal_array("N");

        return TriangleMesh::build_triangles(render_from_object, reverse_orientation, points,
                                             indices, normals, uv, gpu_dynamic_pointers);
    }

    if (type_of_shape == "loopsubdiv") {
        auto levels = parameters.get_integer("levels", 3);
        auto indices = parameters.get_integers("indices");
        auto points = parameters.get_point3_array("P");

        const auto loop_subdivide_data = LoopSubdivide(levels, indices, points);

        return TriangleMesh::build_triangles(render_from_object, reverse_orientation,
                                             loop_subdivide_data.p_limit,
                                             loop_subdivide_data.vertex_indices,
                                             loop_subdivide_data.normals, {}, gpu_dynamic_pointers);
    }

    printf("\nShape `%s` not implemented\n", type_of_shape.c_str());

    REPORT_FATAL_ERROR();
    return {nullptr, 0};
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
        return static_cast<const Disk *>(ptr)->bounds();
    }

    case (Type::sphere): {
        return static_cast<const Sphere *>(ptr)->bounds();
    }

    case (Type::triangle): {
        return static_cast<const Triangle *>(ptr)->bounds();
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
FloatType Shape::area() const {
    switch (type) {
    case (Type::disk): {
        return static_cast<const Disk *>(ptr)->area();
    }

    case (Type::sphere): {
        return static_cast<const Sphere *>(ptr)->area();
    }

    case (Type::triangle): {
        return static_cast<const Triangle *>(ptr)->area();
    }
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

PBRT_CPU_GPU
bool Shape::fast_intersect(const Ray &ray, FloatType t_max) const {
    switch (type) {
    case (Type::disk): {
        return static_cast<const Disk *>(ptr)->fast_intersect(ray, t_max);
    }

    case (Type::triangle): {
        return static_cast<const Triangle *>(ptr)->fast_intersect(ray, t_max);
    }

    case (Type::sphere): {
        return static_cast<const Sphere *>(ptr)->fast_intersect(ray, t_max);
    }
    }

    REPORT_FATAL_ERROR();
    return false;
}

PBRT_CPU_GPU
pbrt::optional<ShapeIntersection> Shape::intersect(const Ray &ray, FloatType t_max) const {
    switch (type) {
    case (Type::disk): {
        return static_cast<const Disk *>(ptr)->intersect(ray, t_max);
    }

    case (Type::triangle): {
        return static_cast<const Triangle *>(ptr)->intersect(ray, t_max);
    }

    case (Type::sphere): {
        return static_cast<const Sphere *>(ptr)->intersect(ray, t_max);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
pbrt::optional<ShapeSample> Shape::sample(const ShapeSampleContext &ctx,
                                               const Point2f &u) const {
    switch (type) {
    case (Type::disk): {
        return static_cast<const Disk *>(ptr)->sample(ctx, u);
    }

    case (Type::sphere): {
        return static_cast<const Sphere *>(ptr)->sample(ctx, u);
    }

    case (Type::triangle): {
        return static_cast<const Triangle *>(ptr)->sample(ctx, u);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

PBRT_CPU_GPU
FloatType Shape::pdf(const ShapeSampleContext &ctx, const Vector3f &wi) const {
    switch (type) {
    case (Type::disk): {
        return static_cast<const Disk *>(ptr)->pdf(ctx, wi);
    }

    case (Type::sphere): {
        return static_cast<const Sphere *>(ptr)->pdf(ctx, wi);
    }

    case (Type::triangle): {
        return static_cast<const Triangle *>(ptr)->pdf(ctx, wi);
    }
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

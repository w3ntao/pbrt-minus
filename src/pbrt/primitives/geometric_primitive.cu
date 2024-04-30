#include "pbrt/primitives/geometric_primitive.h"

#include "pbrt/base/shape.h"
#include "pbrt/base/light.h"

PBRT_CPU_GPU
void GeometricPrimitive::init(const Shape *_shape_ptr, const Material *_material,
                              const Light *_area_light) {
    shape_ptr = _shape_ptr;
    material = _material;
    area_light = _area_light;
}

PBRT_CPU_GPU
Bounds3f GeometricPrimitive::bounds() const {
    return shape_ptr->bounds();
}

PBRT_GPU
bool GeometricPrimitive::fast_intersect(const Ray &ray, FloatType t_max) const {
    return shape_ptr->fast_intersect(ray, t_max);
}

PBRT_GPU
std::optional<ShapeIntersection> GeometricPrimitive::intersect(const Ray &ray,
                                                               FloatType t_max) const {
    auto si = shape_ptr->intersect(ray, t_max);
    if (!si.has_value()) {
        return {};
    }

    si->interaction.set_intersection_properties(material, area_light);
    return si;
}

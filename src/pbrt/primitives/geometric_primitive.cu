#include <pbrt/base/light.h>
#include <pbrt/base/shape.h>
#include <pbrt/primitives/geometric_primitive.h>

PBRT_CPU_GPU
const Material *GeometricPrimitive::get_material() const {
    return material;
}

PBRT_CPU_GPU
Bounds3f GeometricPrimitive::bounds() const {
    return shape_ptr->bounds();
}

PBRT_CPU_GPU
bool GeometricPrimitive::fast_intersect(const Ray &ray, Real t_max) const {
    return shape_ptr->fast_intersect(ray, t_max);
}

PBRT_CPU_GPU
pbrt::optional<ShapeIntersection> GeometricPrimitive::intersect(const Ray &ray, Real t_max) const {
    auto si = shape_ptr->intersect(ray, t_max);
    if (!si.has_value()) {
        return {};
    }

    si->interaction.set_intersection_properties(material, area_light, medium_interface, nullptr);
    return si;
}

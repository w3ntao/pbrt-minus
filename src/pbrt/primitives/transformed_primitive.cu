#include <pbrt/base/primitive.h>
#include <pbrt/primitives/transformed_primitive.h>

PBRT_CPU_GPU
const Material *TransformedPrimitive::get_material() const {
    return primitive->get_material();
}

PBRT_CPU_GPU
Bounds3f TransformedPrimitive::bounds() const {
    return render_from_pritimive(primitive->bounds());
}

PBRT_CPU_GPU
bool TransformedPrimitive::fast_intersect(const Ray &ray, FloatType t_max) const {
    auto inverse_ray = render_from_pritimive.apply_inverse(ray, &t_max);
    return primitive->fast_intersect(inverse_ray, t_max);
}

PBRT_CPU_GPU
pbrt::optional<ShapeIntersection> TransformedPrimitive::intersect(const Ray &ray,
                                                                       FloatType t_max) const {
    // Transform ray to primitive-space and intersect with primitive
    auto inverse_ray = render_from_pritimive.apply_inverse(ray, &t_max);

    auto si = primitive->intersect(inverse_ray, t_max);
    if (!si) {
        return {};
    }

    // Return transformed instance's intersection information
    si->interaction = render_from_pritimive(si->interaction);
    return si;
}

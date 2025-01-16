#include <pbrt/filters/triangle.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>

const TriangleFilter *TriangleFilter::create(const ParameterDictionary &parameters,
                                             GPUMemoryAllocator &allocator) {
    auto xw = parameters.get_float("xradius", 2.f);
    auto yw = parameters.get_float("yradius", 2.f);

    auto triangle_filter = allocator.allocate<TriangleFilter>();

    triangle_filter->radius = Vector2f(xw, yw);

    return triangle_filter;
}

PBRT_CPU_GPU
FilterSample TriangleFilter::sample(const Point2f &u) const {
    return {Point2f(sample_tent(u[0], radius.x), sample_tent(u[1], radius.y)), 1.0};
}

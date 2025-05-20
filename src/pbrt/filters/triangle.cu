#include <pbrt/filters/filter_sampler.h>
#include <pbrt/filters/triangle.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>

TriangleFilter::TriangleFilter(const ParameterDictionary &parameters) : radius(NAN, NAN) {
    auto xw = parameters.get_float("xradius", 2.0f);
    auto yw = parameters.get_float("yradius", 2.0f);

    radius = Vector2f(xw, yw);
}

PBRT_CPU_GPU
FilterSample TriangleFilter::sample(const Point2f &u) const {
    return {Point2f(sample_tent(u[0], radius.x), sample_tent(u[1], radius.y)), 1.0};
}

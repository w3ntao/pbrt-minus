#include <pbrt/filters/box.h>
#include <pbrt/filters/filter_sampler.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>

const BoxFilter *BoxFilter::create(const ParameterDictionary &parameters,
                                   GPUMemoryAllocator &allocator) {
    auto xw = parameters.get_float("xradius", 0.5f);
    auto yw = parameters.get_float("yradius", 0.5f);

    auto box_filter = allocator.allocate<BoxFilter>();

    box_filter->radius = Vector2f(xw, yw);

    return box_filter;
}

PBRT_CPU_GPU
FilterSample BoxFilter::sample(const Point2f u) const {
    Point2f p(pbrt::lerp(u[0], -radius.x, radius.x), pbrt::lerp(u[1], -radius.y, radius.y));
    return FilterSample(p, 1.0);
}

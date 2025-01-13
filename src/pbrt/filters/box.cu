#include <pbrt/filters/box.h>
#include <pbrt/scene/parameter_dictionary.h>

#include <pbrt/gpu/gpu_memory_allocator.h>

const BoxFilter *BoxFilter::create(const ParameterDictionary &parameters,
                                   GPUMemoryAllocator &allocator) {
    auto xw = parameters.get_float("xradius", 0.5f);
    auto yw = parameters.get_float("yradius", 0.5f);

    auto box_filter = allocator.allocate<BoxFilter>();

    box_filter->radius = Vector2f(xw, yw);

    return box_filter;
}

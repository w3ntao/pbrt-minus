#include "pbrt/filters/box.h"
#include "pbrt/scene/parameter_dictionary.h"

const BoxFilter *BoxFilter::create(const ParameterDictionary &parameters,
                                   std::vector<void *> &gpu_dynamic_pointers) {
    auto xw = parameters.get_float("xradius", 0.5f);
    auto yw = parameters.get_float("yradius", 0.5f);

    BoxFilter *box_filter;
    CHECK_CUDA_ERROR(cudaMallocManaged(&box_filter, sizeof(BoxFilter)));
    gpu_dynamic_pointers.push_back(box_filter);

    box_filter->init(Vector2f(xw, yw));

    return box_filter;
}

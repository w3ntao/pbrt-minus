#include "pbrt/base/filter.h"
#include "pbrt/filters/box.h"

const Filter *Filter::create_box_filter(FloatType radius,
                                        std::vector<void *> &gpu_dynamic_pointers) {
    BoxFilter *box_filter;
    CHECK_CUDA_ERROR(cudaMallocManaged(&box_filter, sizeof(BoxFilter)));

    Filter *filter;
    CHECK_CUDA_ERROR(cudaMallocManaged(&filter, sizeof(Filter)));

    gpu_dynamic_pointers.push_back(box_filter);
    gpu_dynamic_pointers.push_back(filter);

    box_filter->init(radius);
    filter->init(box_filter);

    return filter;
}

void Filter::init(const BoxFilter *box_filter) {
    ptr = box_filter;
    type = Type::box;
}

PBRT_CPU_GPU
FilterSample Filter::sample(Point2f u) const {
    switch (type) {
    case (Type::box): {
        return ((BoxFilter *)ptr)->sample(u);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

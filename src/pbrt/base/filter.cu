#include "pbrt/base/filter.h"
#include "pbrt/filters/box.h"

void Filter::init(BoxFilter *box_filter) {
    filter_ptr = box_filter;
    filter_type = FilterType::box;
}

PBRT_CPU_GPU
FilterSample Filter::sample(Point2f u) const {
    switch (filter_type) {
    case (FilterType::box): {
        return ((BoxFilter *)filter_ptr)->sample(u);
    }
    }

    report_function_error_and_exit(__func__);
    return {};
}

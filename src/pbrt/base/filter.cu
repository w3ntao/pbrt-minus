#include "pbrt/base/filter.h"
#include "pbrt/filters/box.h"

void Filter::init(const BoxFilter *box_filter) {
    filter_ptr = box_filter;
    filter_type = Type::box;
}

PBRT_CPU_GPU
FilterSample Filter::sample(Point2f u) const {
    switch (filter_type) {
    case (Type::box): {
        return ((BoxFilter *)filter_ptr)->sample(u);
    }
    }

    REPORT_FATAL_ERROR();
    return {};
}

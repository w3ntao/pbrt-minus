#include "pbrt/base/filter.h"
#include "pbrt/filters/box.h"

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

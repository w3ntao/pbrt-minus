#include <pbrt/filters/box.h>
#include <pbrt/filters/filter_sampler.h>
#include <pbrt/scene/parameter_dictionary.h>

BoxFilter::BoxFilter(const ParameterDictionary &parameters) : radius(NAN, NAN) {
    radius.x = parameters.get_float("xradius", 0.5f);
    radius.y = parameters.get_float("yradius", 0.5f);
}

PBRT_CPU_GPU
FilterSample BoxFilter::sample(const Point2f u) const {
    Point2f p(pbrt::lerp(u[0], -radius.x, radius.x), pbrt::lerp(u[1], -radius.y, radius.y));
    return FilterSample(p, 1.0);
}

#include "pbrt/base/sampler.h"
#include "pbrt/base/filter.h"

PBRT_GPU
CameraSample Sampler::get_camera_sample(const Point2i &pPixel, const Filter *filter) {
    auto fs = filter->sample(get_pixel_2d());

    return CameraSample(pPixel.to_point2f() + fs.p + Vector2f(0.5, 0.5), get_2d(), fs.weight);
}

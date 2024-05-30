#include "pbrt/base/light.h"
#include "pbrt/lights/diffuse_area_light.h"
#include "pbrt/lights/image_infinite_light.h"

PBRT_CPU_GPU
void Light::init(DiffuseAreaLight *diffuse_area_light) {
    type = Type::diffuse_area_light;
    ptr = diffuse_area_light;
}

PBRT_CPU_GPU
void Light::init(ImageInfiniteLight *image_infinite_light) {
    type = Type::image_infinite_light;
    ptr = image_infinite_light;
}

PBRT_GPU
SampledSpectrum Light::l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                         const SampledWavelengths &lambda) const {
    switch (type) {
    case (Type::diffuse_area_light): {
        return ((DiffuseAreaLight *)ptr)->l(p, n, uv, w, lambda);
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_GPU
SampledSpectrum Light::le(const Ray &ray, const SampledWavelengths &lambda) const {
    switch (type) {
    case (Type::image_infinite_light): {
        return ((ImageInfiniteLight *)ptr)->le(ray, lambda);
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_GPU
cuda::std::optional<LightLiSample> Light::sample_li(const LightSampleContext ctx, const Point2f u,
                                                    SampledWavelengths &lambda) const {
    switch (type) {
    case (Type::diffuse_area_light): {
        return ((DiffuseAreaLight *)ptr)->sample_li(ctx, u, lambda);
    }
    case (Type::image_infinite_light): {
        return ((ImageInfiniteLight *)ptr)->sample_li(ctx, u, lambda);
    }
    }

    REPORT_FATAL_ERROR();
}

void Light::preprocess(const Bounds3<FloatType> &scene_bounds) {
    switch (type) {
    case (Type::diffuse_area_light): {
        // do nothing
        return;
    }
    case (Type::image_infinite_light): {
        return ((ImageInfiniteLight *)ptr)->preprocess(scene_bounds);
    }
    }

    REPORT_FATAL_ERROR();
}

#include "pbrt/base/light.h"
#include "pbrt/lights/diffuse_area_light.h"

PBRT_CPU_GPU
void Light::init(const DiffuseAreaLight *diffuse_area_light) {
    light_type = Type::diffuse_area_light;
    light_ptr = diffuse_area_light;
}

PBRT_GPU
SampledSpectrum Light::l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                         const SampledWavelengths &lambda) const {
    switch (light_type) {
    case (Type::diffuse_area_light): {
        return ((DiffuseAreaLight *)light_ptr)->l(p, n, uv, w, lambda);
    }
    }

    REPORT_FATAL_ERROR();
}

PBRT_GPU
cuda::std::optional<LightLiSample> Light::sample_li(const LightSampleContext ctx, const Point2f u,
                                                    SampledWavelengths &lambda,
                                                    bool allow_incomplete_pdf) const {
    switch (light_type) {
    case (Type::diffuse_area_light): {
        return ((DiffuseAreaLight *)light_ptr)->sample_li(ctx, u, lambda, allow_incomplete_pdf);
    }
    }

    REPORT_FATAL_ERROR();
}

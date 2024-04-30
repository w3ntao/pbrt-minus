#include "pbrt/base/light.h"
#include "pbrt/lights/diffuse_area_light.h"

void Light::init(const DiffuseAreaLight *diffuse_area_light) {
    light_type = _LightType::diffuse_area_light;
    light_ptr = (void *)diffuse_area_light;
}

PBRT_GPU
SampledSpectrum Light::l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                         const SampledWavelengths &lambda) const {
    switch (light_type) {
    case (_LightType::diffuse_area_light): {
        return ((DiffuseAreaLight *)light_ptr)->l(p, n, uv, w, lambda);
    }
    }

    REPORT_FATAL_ERROR();
}

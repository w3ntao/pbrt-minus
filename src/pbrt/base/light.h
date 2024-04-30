#pragma once

#include "pbrt/euclidean_space/transform.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"
#include "pbrt/spectrum_util/sampled_wavelengths.h"

enum class LightType {
    delta_position,
    delta_direction,
    area,
    infinite,
};

struct LightBase {
    LightType light_type;
    Transform render_from_light;
};

class DiffuseAreaLight;

class Light {
  public:
    enum class _LightType {
        diffuse_area_light,
    };

    void init(const DiffuseAreaLight *diffuse_area_light);

    PBRT_CPU_GPU _LightType get_light_type() const {
        return light_type;
    }

    PBRT_GPU
    SampledSpectrum l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                      const SampledWavelengths &lambda) const;

  private:
    _LightType light_type;
    void *light_ptr;
};

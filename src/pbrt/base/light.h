#pragma once

#include <cuda/std/optional>

#include "pbrt/base/interaction.h"
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

class Light;
class DiffuseAreaLight;

struct SampledLight {
    const Light *light;
    FloatType p;
};

struct LightLiSample {
    SampledSpectrum l;
    Vector3f wi;
    FloatType pdf;
    Interaction p_light;

    PBRT_CPU_GPU
    LightLiSample(SampledSpectrum _l, Vector3f _wi, FloatType _pdf, Interaction _p_light)
        : l(_l), wi(_wi), pdf(_pdf), p_light(_p_light) {}
};

struct LightSampleContext {
    Point3fi pi;
    Normal3f n;
    Normal3f ns;

    PBRT_CPU_GPU
    LightSampleContext(const SurfaceInteraction si) : pi(si.pi), n(si.n), ns(si.shading.n) {}

    PBRT_CPU_GPU
    Point3f p() const {
        return pi.to_point3f();
    }
};

class Light {
  public:
    enum class Type {
        diffuse_area_light,
    };

    PBRT_CPU_GPU
    void init(const DiffuseAreaLight *diffuse_area_light);

    PBRT_CPU_GPU Type get_light_type() const {
        return light_type;
    }

    PBRT_GPU
    SampledSpectrum l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                      const SampledWavelengths &lambda) const;

    PBRT_GPU
    cuda::std::optional<LightLiSample> sample_li(const LightSampleContext ctx, const Point2f u,
                                                 SampledWavelengths &lambda,
                                                 bool allow_incomplete_pdf) const;

  public:
    Type light_type;
    const void *light_ptr;
};

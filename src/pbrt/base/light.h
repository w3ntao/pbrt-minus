#pragma once

#include <vector>
#include <cuda/std/optional>

#include "pbrt/base/interaction.h"
#include "pbrt/euclidean_space/transform.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"
#include "pbrt/spectrum_util/sampled_wavelengths.h"

template <typename T>
class Bounds3;

class DiffuseAreaLight;
class DistantLight;
class GlobalSpectra;
class ImageInfiniteLight;
class Light;
class Shape;
class ParameterDictionary;

enum class LightType {
    delta_position,
    delta_direction,
    area,
    infinite,
};

// Light Inline Functions
PBRT_CPU_GPU inline bool is_deltaLight(LightType type) {
    return type == LightType::delta_position || type == LightType::delta_direction;
}

struct LightBase {
    LightType light_type;
    Transform render_from_light;

    PBRT_CPU_GPU
    LightType get_light_type() const {
        return light_type;
    }
};

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
    LightSampleContext()
        : pi(Point3fi()), n(Normal3f(NAN, NAN, NAN)), ns(Normal3f(NAN, NAN, NAN)) {}

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
        distant_light,
        image_infinite_light,
    };

    static Light *create(const std::string &type_of_light, const Transform &render_from_light,
                         const ParameterDictionary &parameters,
                         std::vector<void *> &gpu_dynamic_pointers);

    static Light *create_diffuse_area_lights(const Shape *shapes, const uint num,
                                             const Transform &render_from_light,
                                             const ParameterDictionary &parameters,
                                             std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU
    void init(DistantLight *distant_light);

    PBRT_CPU_GPU
    void init(DiffuseAreaLight *diffuse_area_light);

    PBRT_CPU_GPU
    void init(ImageInfiniteLight *image_infinite_light);

    PBRT_CPU_GPU
    LightType get_light_type() const;

    PBRT_GPU
    SampledSpectrum l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                      const SampledWavelengths &lambda) const;

    PBRT_GPU
    SampledSpectrum le(const Ray &ray, const SampledWavelengths &lambda) const;

    PBRT_GPU
    cuda::std::optional<LightLiSample> sample_li(const LightSampleContext &ctx, const Point2f &u,
                                                 SampledWavelengths &lambda) const;

    PBRT_GPU
    FloatType pdf_li(const LightSampleContext &ctx, const Vector3f &wi,
                     bool allow_incomplete_pdf = false) const;

    void preprocess(const Bounds3<FloatType> &scene_bounds);

  public:
    Type type;
    void *ptr;
};

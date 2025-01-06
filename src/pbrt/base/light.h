#pragma once

#include "pbrt/base/interaction.h"
#include "pbrt/euclidean_space/transform.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"
#include "pbrt/spectrum_util/sampled_wavelengths.h"
#include <cuda/std/optional>
#include <vector>

template <typename T>
class Bounds3;

class DiffuseAreaLight;
class DistantLight;
class GlobalSpectra;
class ImageInfiniteLight;
class Light;
class Shape;
class SpotLight;
class ParameterDictionary;
class UniformInfiniteLight;

enum class LightType {
    delta_position,
    delta_direction,
    area,
    infinite,
};

namespace pbrt {
// Light Inline Functions
PBRT_CPU_GPU inline bool is_delta_light(LightType type) {
    return type == LightType::delta_position || type == LightType::delta_direction;
}
} // namespace pbrt

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
    LightSampleContext(const SurfaceInteraction &si) : pi(si.pi), n(si.n), ns(si.shading.n) {}

    PBRT_CPU_GPU
    LightSampleContext(const Interaction &intr)
        : pi(intr.pi), n(Normal3f(NAN, NAN, NAN)), ns(Normal3f(NAN, NAN, NAN)) {}

    PBRT_CPU_GPU
    Point3f p() const {
        return pi.to_point3f();
    }
};

// LightLeSample Definition
struct LightLeSample {
    // LightLeSample Public Methods

    PBRT_CPU_GPU
    LightLeSample() : pdfPos(0), pdfDir(0) {}

    PBRT_CPU_GPU
    LightLeSample(const SampledSpectrum &L, const Ray &ray, FloatType pdfPos, FloatType pdfDir)
        : L(L), ray(ray), pdfPos(pdfPos), pdfDir(pdfDir) {}
    PBRT_CPU_GPU
    LightLeSample(const SampledSpectrum &_L, const Ray &_ray, const Interaction &_intr,
                  FloatType _pdfPos, FloatType _pdfDir)
        : L(_L), ray(_ray), intr(_intr), pdfPos(_pdfPos), pdfDir(_pdfDir) {
        if (DEBUG_MODE && this->intr->n != Normal3f(0, 0, 0)) {
            REPORT_FATAL_ERROR();
        }
    }

    PBRT_CPU_GPU
    FloatType abs_cos_theta(const Vector3f w) const {
        return intr ? intr->n.abs_dot(w) : 1;
    }

    SampledSpectrum L;
    Ray ray;
    cuda::std::optional<Interaction> intr;
    FloatType pdfPos;
    FloatType pdfDir;
};

class Light {
  public:
    enum class Type {
        diffuse_area_light,
        distant_light,
        image_infinite_light,
        spot_light,
        uniform_infinite_light,
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
    void init(SpotLight *spot_light);

    PBRT_CPU_GPU
    void init(UniformInfiniteLight *uniform_infinite_light);

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
    cuda::std::optional<LightLeSample> sample_le(const Point2f u1, const Point2f u2,
                                                 SampledWavelengths &lambda) const;

    PBRT_GPU
    FloatType pdf_li(const LightSampleContext &ctx, const Vector3f &wi,
                     bool allow_incomplete_pdf = false) const;

    PBRT_GPU
    void pdf_le(const Interaction &intr, Vector3f w, FloatType *pdfPos, FloatType *pdfDir) const;

    PBRT_GPU
    void pdf_le(const Ray &ray, FloatType *pdfPos, FloatType *pdfDir) const;

    PBRT_CPU_GPU
    SampledSpectrum phi(const SampledWavelengths &lambda) const;

    void preprocess(const Bounds3<FloatType> &scene_bounds);

    Type type;

  private:
    void *ptr;
};

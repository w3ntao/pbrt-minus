#pragma once

#include <pbrt/base/light.h>
#include <pbrt/euclidean_space/point3.h>
#include <pbrt/gpu/macro.h>

class GPUMemoryAllocator;
class ParameterDictionary;
class Spectrum;
class Transform;

class SpotLight : public LightBase {
  public:
    SpotLight(const Transform &renderFromLight, const Spectrum *Iemit, Real _scale, Real totalWidth,
              Real falloffStart)
        : LightBase(LightType::delta_position, renderFromLight) {
        i_emit = Iemit;
        scale = _scale;

        cosFalloffEnd = std::cos(degree_to_radian(totalWidth));
        cosFalloffStart = std::cos(degree_to_radian(falloffStart));
    }

    static SpotLight *create(const Transform &renderFromLight,
                             const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    void preprocess(const Bounds3f &scene_bounds) {
        REPORT_FATAL_ERROR();
    }

    PBRT_CPU_GPU
    SampledSpectrum l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                      const SampledWavelengths &lambda) const {
        REPORT_FATAL_ERROR();
        return {};
    }

    PBRT_CPU_GPU
    pbrt::optional<LightLiSample> sample_li(const LightSampleContext &ctx, const Point2f &u,
                                            SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pbrt::optional<LightLeSample> sample_le(const Point2f u1, const Point2f u2,
                                            SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    void pdf_le(const Ray &ray, Real *pdfPos, Real *pdfDir) const;

    PBRT_CPU_GPU
    Real pdf_li(const LightSampleContext &ctx, const Vector3f &wi,
                bool allow_incomplete_pdf) const {
        return 0.0;
    }

    PBRT_CPU_GPU
    SampledSpectrum phi(const SampledWavelengths &lambda) const;

  private:
    const Spectrum *i_emit = nullptr;
    Real scale = NAN;
    Real cosFalloffStart = NAN;
    Real cosFalloffEnd = NAN;

    PBRT_CPU_GPU
    SampledSpectrum I(const Vector3f &w, const SampledWavelengths &lambda) const;
};

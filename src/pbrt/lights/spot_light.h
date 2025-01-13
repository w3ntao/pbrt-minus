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
    static SpotLight *create(const Transform &renderFromLight,
                             const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    void preprocess(const Bounds3f &scene_bounds) {
        REPORT_FATAL_ERROR();
    }

    PBRT_CPU_GPU
    SampledSpectrum l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                      const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pbrt::optional<LightLiSample> sample_li(const LightSampleContext &ctx, const Point2f &u,
                                            SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    pbrt::optional<LightLeSample> sample_le(const Point2f u1, const Point2f u2,
                                            SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    void pdf_le(const Ray &ray, FloatType *pdfPos, FloatType *pdfDir) const;

    PBRT_CPU_GPU
    FloatType pdf_li(const LightSampleContext &ctx, const Vector3f &wi,
                     bool allow_incomplete_pdf = false) const;
    PBRT_CPU_GPU
    SampledSpectrum phi(const SampledWavelengths &lambda) const;

  private:
    const Spectrum *i_emit;
    FloatType scale;
    FloatType cosFalloffStart;
    FloatType cosFalloffEnd;

    void init(const Transform &renderFromLight, const Spectrum *Iemit, FloatType _scale,
              FloatType totalWidth, FloatType falloffStart);

    PBRT_CPU_GPU
    SampledSpectrum I(const Vector3f &w, const SampledWavelengths &lambda) const;
};

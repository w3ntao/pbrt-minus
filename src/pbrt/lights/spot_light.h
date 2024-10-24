#pragma once

#include "pbrt/base/light.h"
#include "pbrt/euclidean_space/point3.h"
#include "pbrt/util/macro.h"
#include <vector>

class ParameterDictionary;
class Spectrum;
class Transform;

class SpotLight : public LightBase {
  public:
    static SpotLight *create(const Transform &renderFromLight,
                             const ParameterDictionary &parameters,
                             std::vector<void *> &gpu_dynamic_pointers);

    void preprocess(const Bounds3f &scene_bounds) {
        REPORT_FATAL_ERROR();
    }

    PBRT_GPU
    SampledSpectrum l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                      const SampledWavelengths &lambda) const;

    PBRT_GPU
    cuda::std::optional<LightLiSample> sample_li(const LightSampleContext &ctx, const Point2f &u,
                                                 SampledWavelengths &lambda) const;

    PBRT_GPU
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

    PBRT_GPU
    SampledSpectrum I(const Vector3f &w, const SampledWavelengths &lambda) const;
};

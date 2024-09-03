#pragma once

#include "pbrt/base/light.h"
#include <cuda/std/optional>

class GlobalSpectra;
class ParameterDictionary;
class RGBColorSpace;
class Spectrum;
class Shape;

class DiffuseAreaLight : public LightBase {
  public:
    void init(const Shape *_shape, const Transform &_render_from_light,
              const ParameterDictionary &parameters, std::vector<void *> &gpu_dynamic_pointers);

    PBRT_GPU
    SampledSpectrum l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                      const SampledWavelengths &lambda) const;

    PBRT_GPU
    cuda::std::optional<LightLiSample> sample_li(const LightSampleContext &ctx, const Point2f &u,
                                                 SampledWavelengths &lambda) const;

    PBRT_GPU
    FloatType pdf_li(const LightSampleContext &ctx, const Vector3f &wi,
                     bool allow_incomplete_pdf) const;

    PBRT_CPU_GPU
    SampledSpectrum phi(const SampledWavelengths &lambda) const;

  private:
    const Shape *shape;
    bool two_sided;
    const Spectrum *l_emit;
    FloatType scale;
};

#pragma once

#include <cuda/std/optional>

#include "pbrt/base/light.h"
#include "pbrt/spectra/densely_sampled_spectrum.h"

class GlobalSpectra;
class ParameterDictionary;
class RGBColorSpace;
class Spectrum;
class Shape;

class DiffuseAreaLight : public LightBase {
  public:
    void init(const Shape *_shape, const Transform &_render_from_light,
              const ParameterDictionary &parameters);

    PBRT_GPU
    SampledSpectrum l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                      const SampledWavelengths &lambda) const;

    PBRT_GPU
    cuda::std::optional<LightLiSample> sample_li(const LightSampleContext &ctx, const Point2f &u,
                                                 SampledWavelengths &lambda) const;

    PBRT_GPU
    FloatType pdf_li(const LightSampleContext &ctx, const Vector3f &wi,
                     bool allow_incomplete_pdf) const;

  private:
    const Shape *shape;
    bool two_sided;
    DenselySampledSpectrum l_emit;
    FloatType scale;
};

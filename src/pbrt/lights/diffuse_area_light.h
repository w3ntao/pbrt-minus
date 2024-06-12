#pragma once

#include <cuda/std/optional>

#include "pbrt/base/light.h"
#include "pbrt/spectra/densely_sampled_spectrum.h"

class ParameterDictionary;
class RGBColorSpace;
class Spectrum;
class Shape;

namespace GPU {
class GlobalVariable;
}

class DiffuseAreaLight : public LightBase {
  public:
    void init(const Transform &_render_from_light, const ParameterDictionary &parameters,
              const Shape *_shape, const GPU::GlobalVariable *global_variable);

    PBRT_GPU
    SampledSpectrum l(Point3f p, Normal3f n, Point2f uv, Vector3f w,
                      const SampledWavelengths &lambda) const;

    PBRT_GPU
    cuda::std::optional<LightLiSample> sample_li(const LightSampleContext &ctx, const Point2f &u,
                                                 SampledWavelengths &lambda) const;

  private:
    const Shape *shape;
    bool two_sided;
    DenselySampledSpectrum l_emit;
    FloatType scale;
};

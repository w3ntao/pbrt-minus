#pragma once

#include "pbrt/base/bsdf.h"

class MaterialEvalContext;
class SpectrumTexture;
class SampledWavelengths;
class DiffuseBxDF;

class DiffuseMaterial {
  public:
    void init(const SpectrumTexture *_reflectance);

    PBRT_GPU
    DiffuseBxDF get_diffuse_bsdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *reflectance;
};

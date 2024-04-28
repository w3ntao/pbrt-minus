#pragma once

#include "pbrt/base/bsdf.h"

class MaterialEvalContext;
class SpectrumConstantTexture;
class SampledWavelengths;
class BSDF;
class DiffuseBxDF;

class DiffuseMaterial {
  public:
    void init(const SpectrumConstantTexture *_reflectance);

    PBRT_GPU
    DiffuseBxDF get_diffuse_bsdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

  private:
    const SpectrumConstantTexture *reflectance = nullptr;
};

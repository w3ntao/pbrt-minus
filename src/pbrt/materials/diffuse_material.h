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
    void get_diffuse_bsdf(BSDF &bsdf, const MaterialEvalContext &ctx,
                          SampledWavelengths &lambda) const;

    // TODO: make this private
  public:
    const SpectrumConstantTexture *reflectance = nullptr;
};

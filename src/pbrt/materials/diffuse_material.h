#pragma once

#include <vector>
#include <map>

#include "pbrt/base/bsdf.h"

class MaterialEvalContext;
class SpectrumTexture;
class SampledWavelengths;
class DiffuseBxDF;
class ParameterDictionary;
class RGBColorSpace;
class SpectrumTexture;

class DiffuseMaterial {
  public:
    static const DiffuseMaterial *create(const SpectrumTexture *_reflectance,
                                         std::vector<void *> &gpu_dynamic_pointers);

    void init(const ParameterDictionary &parameters, std::vector<void *> &gpu_dynamic_pointers);

    PBRT_GPU
    DiffuseBxDF get_diffuse_bsdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *reflectance;
};

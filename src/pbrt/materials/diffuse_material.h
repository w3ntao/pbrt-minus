#pragma once

#include "pbrt/base/bsdf.h"
#include <map>
#include <vector>

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

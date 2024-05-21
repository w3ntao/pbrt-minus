#pragma once

#include <vector>
#include <map>

#include "pbrt/base/bsdf.h"

class MaterialEvalContext;
class SpectrumTexture;
class SampledWavelengths;
class DiffuseBxDF;
class ParameterDict;
class RGBColorSpace;
class SpectrumTexture;

class DiffuseMaterial {
  public:
    void init(const RGBColorSpace *color_space, const ParameterDict &parameters,
              std::vector<void *> &gpu_dynamic_pointers);

    void init_reflectance(const SpectrumTexture *_reflectance);

    PBRT_GPU
    DiffuseBxDF get_diffuse_bsdf(const MaterialEvalContext &ctx, SampledWavelengths &lambda) const;

  private:
    const SpectrumTexture *reflectance;
};

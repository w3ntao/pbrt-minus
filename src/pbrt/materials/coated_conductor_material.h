#pragma once

#include <vector>

class ParameterDictionary;

class FloatTexture;
class Spectrum;
class SpectrumTexture;

class CoatedConductorMaterial {
  public:
    void init(const ParameterDictionary &parameters, std::vector<void *> &gpu_dynamic_pointers);

  private:
    // TODO: renmae variables;

    const FloatTexture *interfaceURoughness, *interfaceVRoughness, *thickness;
    const Spectrum *interfaceEta;
    const FloatTexture *g;
    const SpectrumTexture *albedo;
    const FloatTexture *conductorURoughness, *conductorVRoughness;
    const SpectrumTexture *conductorEta, *k, *reflectance;
    bool remapRoughness;
    int maxDepth, nSamples;
};

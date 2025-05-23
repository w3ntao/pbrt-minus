#pragma once

#include <pbrt/base/spectrum.h>

class ParameterDictionary;
class Transform;
struct TextureEvalContext;

class SpectrumConstantTexture {
  public:
    SpectrumConstantTexture(const ParameterDictionary &parameters, SpectrumType spectrum_type,
                            GPUMemoryAllocator &allocator);

    SpectrumConstantTexture(const Spectrum *_value) : value(_value) {}

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    const Spectrum *value = nullptr;
};

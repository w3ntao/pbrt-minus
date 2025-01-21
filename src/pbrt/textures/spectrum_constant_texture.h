#pragma once

#include <pbrt/base/spectrum.h>

class ParameterDictionary;
class Transform;
struct TextureEvalContext;

class SpectrumConstantTexture {
  public:
    static const SpectrumConstantTexture *create(const ParameterDictionary &parameters,
                                                 SpectrumType spectrum_type,
                                                 GPUMemoryAllocator &allocator);

    void init(const Spectrum *_value) {
        value = _value;
    }

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    const Spectrum *value;
};

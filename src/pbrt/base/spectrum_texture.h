#pragma once

#include <vector>

#include "pbrt/util/macro.h"

#include "texture_eval_context.h"

class ParameterDictionary;

class RGBColorSpace;

class Spectrum;
class SpectrumConstantTexture;
class SpectrumImageTexture;
class SpectrumScaleTexture;

class SpectrumTexture {
  public:
    enum class Type {
        constant,
        image,
        scale,
    };

    static const SpectrumTexture *create(const std::string &texture_type,
                                         const ParameterDictionary &parameters,
                                         const RGBColorSpace *color_space,
                                         std::vector<void *> &gpu_dynamic_pointers);

    static const SpectrumTexture *
    create_constant_float_val_texture(FloatType val, std::vector<void *> &gpu_dynamic_pointers);

    static const SpectrumTexture *
    create_constant_texture(const Spectrum *spectrum, std::vector<void *> &gpu_dynamic_pointers);

    void init(const SpectrumConstantTexture *constant_texture);

    void init(const SpectrumImageTexture *image_texture);

    void init(const SpectrumScaleTexture *scale_texture);

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    Type type;
    const void *ptr;
};

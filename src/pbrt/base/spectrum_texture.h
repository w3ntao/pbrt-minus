#pragma once

#include "pbrt/base/spectrum.h"
#include "pbrt/base/texture_eval_context.h"
#include "pbrt/util/macro.h"
#include <vector>

class Transform;
class ParameterDictionary;
class RGBColorSpace;

class Spectrum;
class SpectrumConstantTexture;
class SpectrumImageTexture;
class SpectrumScaledTexture;

class SpectrumTexture {
  public:
    enum class Type {
        constant,
        image,
        scaled,
    };

    static const SpectrumTexture *
    create(const std::string &texture_type, SpectrumType spectrum_type,
           const Transform &render_from_object, const RGBColorSpace *color_space,
           const ParameterDictionary &parameters, std::vector<void *> &gpu_dynamic_pointers);

    static const SpectrumTexture *
    create_constant_float_val_texture(FloatType val, std::vector<void *> &gpu_dynamic_pointers);

    static const SpectrumTexture *
    create_constant_texture(const Spectrum *spectrum, std::vector<void *> &gpu_dynamic_pointers);

    void init(const SpectrumConstantTexture *constant_texture);

    void init(const SpectrumImageTexture *image_texture);

    void init(const SpectrumScaledTexture *scale_texture);

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    Type type;
    const void *ptr;
};

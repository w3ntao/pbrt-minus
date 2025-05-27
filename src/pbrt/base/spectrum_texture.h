#pragma once

#include <pbrt/base/spectrum.h>
#include <pbrt/gpu/macro.h>

class SpectrumCheckerboardTexture;
class SpectrumDirectionMixTexture;
class SpectrumConstantTexture;
class SpectrumImageTexture;
class SpectrumMixTexture;
class SpectrumScaledTexture;

class GPUMemoryAllocator;
class RGBColorSpace;
class Transform;
class ParameterDictionary;
class TextureEvalContext;

class SpectrumTexture {
  public:
    enum class Type {
        checkerboard,
        constant,
        direction_mix,
        image,
        mix,
        scaled,
    };

    static const SpectrumTexture *
    create(const std::string &texture_type, SpectrumType spectrum_type,
           const Transform &render_from_texture, const RGBColorSpace *color_space,
           const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    static const SpectrumTexture *create_constant_float_val_texture(Real val,
                                                                    GPUMemoryAllocator &allocator);

    static const SpectrumTexture *create_constant_texture(const Spectrum *spectrum,
                                                          GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    Type type;
    const void *ptr = nullptr;

    void init(const SpectrumCheckerboardTexture *checkerboard_texture);

    void init(const SpectrumConstantTexture *constant_texture);

    void init(const SpectrumDirectionMixTexture *direction_mix_texture);

    void init(const SpectrumImageTexture *image_texture);

    void init(const SpectrumMixTexture *mix_texture);

    void init(const SpectrumScaledTexture *scale_texture);
};

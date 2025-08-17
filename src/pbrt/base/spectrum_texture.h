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

struct TextureEvalContext;

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

    explicit SpectrumTexture(const SpectrumCheckerboardTexture *checkerboard_texture)
        : type(Type::checkerboard), ptr(checkerboard_texture) {}

    explicit SpectrumTexture(const SpectrumConstantTexture *constant_texture)
        : type(Type::constant), ptr(constant_texture) {}

    explicit SpectrumTexture(const SpectrumDirectionMixTexture *direction_mix_texture)
        : type(Type::direction_mix), ptr(direction_mix_texture) {}

    explicit SpectrumTexture(const SpectrumImageTexture *image_texture)
        : type(Type::image), ptr(image_texture) {}

    explicit SpectrumTexture(const SpectrumMixTexture *mix_texture)
        : type(Type::mix), ptr(mix_texture) {}

    explicit SpectrumTexture(const SpectrumScaledTexture *scale_texture)
        : type(Type::scaled), ptr(scale_texture) {}

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
};

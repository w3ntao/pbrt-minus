#pragma once

#include <cuda/std/variant>
#include <pbrt/textures/spectrum_checkerboard_texture.h>
#include <pbrt/textures/spectrum_constant_texture.h>
#include <pbrt/textures/spectrum_direction_mix_texture.h>
#include <pbrt/textures/spectrum_image_texture.h>
#include <pbrt/textures/spectrum_mix_texture.h>
#include <pbrt/textures/spectrum_scaled_texture.h>

namespace HIDDEN {
using SpectrumTextureVariants =
    cuda::std::variant<SpectrumCheckerboardTexture, SpectrumDirectionMixTexture,
                       SpectrumConstantTexture, SpectrumImageTexture, SpectrumMixTexture,
                       SpectrumScaledTexture>;
}

class SpectrumTexture : public HIDDEN::SpectrumTextureVariants {
    using HIDDEN::SpectrumTextureVariants::SpectrumTextureVariants;

  public:
    static const SpectrumTexture *
    create(const std::string &texture_type, SpectrumType spectrum_type,
           const Transform &render_from_texture, const RGBColorSpace *color_space,
           const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    static const SpectrumTexture *create_constant_float_val_texture(Real val,
                                                                    GPUMemoryAllocator &allocator);

    static const SpectrumTexture *create_constant_texture(const Spectrum *spectrum,
                                                          GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx,
                             const SampledWavelengths &lambda) const {
        return cuda::std::visit([&](auto &x) { return x.evaluate(ctx, lambda); }, *this);
    }
};

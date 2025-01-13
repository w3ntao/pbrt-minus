#include <pbrt/base/spectrum_texture.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/textures/spectrum_scaled_texture.h>

void SpectrumScaledTexture::init(SpectrumType spectrum_type, const ParameterDictionary &parameters,
                                 GPUMemoryAllocator &allocator) {
    texture = parameters.get_spectrum_texture("tex", spectrum_type, allocator);

    scale = parameters.get_float("scale", 1.0);
}

PBRT_CPU_GPU
SampledSpectrum SpectrumScaledTexture::evaluate(const TextureEvalContext &ctx,
                                                const SampledWavelengths &lambda) const {
    if (scale == 0.0) {
        return 0.0;
    }

    return texture->evaluate(ctx, lambda) * scale;
}

#include <pbrt/base/float_texture.h>
#include <pbrt/base/spectrum_texture.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/textures/spectrum_mix_texture.h>

const SpectrumMixTexture *SpectrumMixTexture::create(const ParameterDictionary &parameters,
                                                     const SpectrumType spectrum_type,
                                                     GPUMemoryAllocator &allocator) {
    auto tex1 = parameters.get_spectrum_texture("tex1", spectrum_type, allocator);
    if (tex1 == nullptr) {
        tex1 = SpectrumTexture::create_constant_float_val_texture(0.0, allocator);
    }

    auto tex2 = parameters.get_spectrum_texture("tex2", spectrum_type, allocator);
    if (tex2 == nullptr) {
        tex2 = SpectrumTexture::create_constant_float_val_texture(1.0, allocator);
    }

    const auto amount = parameters.get_float_texture("amount", 0.5, allocator);

    auto texture = allocator.allocate<SpectrumMixTexture>();
    texture->tex1 = tex1;
    texture->tex2 = tex2;
    texture->amount = amount;

    return texture;
}

PBRT_CPU_GPU
SampledSpectrum SpectrumMixTexture::evaluate(const TextureEvalContext &ctx,
                                             const SampledWavelengths &lambda) const {
    auto amt = amount->evaluate(ctx);
    SampledSpectrum t1, t2;
    if (amt != 1) {
        t1 = tex1->evaluate(ctx, lambda);
    }
    if (amt != 0) {
        t2 = tex2->evaluate(ctx, lambda);
    }

    return (1 - amt) * t1 + amt * t2;
}

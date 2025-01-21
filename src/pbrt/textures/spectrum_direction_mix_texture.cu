#include <pbrt/base/spectrum_texture.h>
#include <pbrt/base/texture_eval_context.h>
#include <pbrt/euclidean_space/transform.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/textures/spectrum_direction_mix_texture.h>

const SpectrumDirectionMixTexture *SpectrumDirectionMixTexture::create(
    const Transform &render_from_texture, const ParameterDictionary &parameters,
    const SpectrumType spectrumType, GPUMemoryAllocator &allocator) {
    auto tex1 = parameters.get_spectrum_texture("tex1", spectrumType, allocator);
    if (tex1 == nullptr) {
        tex1 = SpectrumTexture::create_constant_float_val_texture(0.0, allocator);
    }

    auto tex2 = parameters.get_spectrum_texture("tex2", spectrumType, allocator);
    if (tex2 == nullptr) {
        tex2 = SpectrumTexture::create_constant_float_val_texture(1.0, allocator);
    }

    const auto dir = parameters.get_vector3f("dir", Vector3f(0, 1, 0));

    auto direction_mix_texture = allocator.allocate<SpectrumDirectionMixTexture>();

    direction_mix_texture->tex1 = tex1;
    direction_mix_texture->tex2 = tex2;
    direction_mix_texture->dir = render_from_texture(dir).normalize();

    return direction_mix_texture;
}

PBRT_CPU_GPU
SampledSpectrum SpectrumDirectionMixTexture::evaluate(const TextureEvalContext &ctx,
                                                      const SampledWavelengths &lambda) const {
    const auto amt = ctx.n.abs_dot(dir);
    SampledSpectrum t1, t2;
    if (amt != 0) {
        t1 = tex1->evaluate(ctx, lambda);
    }
    if (amt != 1) {
        t2 = tex2->evaluate(ctx, lambda);
    }

    return amt * t1 + (1 - amt) * t2;
}

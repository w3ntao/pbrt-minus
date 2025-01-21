#include <pbrt/base/texture_eval_context.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/textures/spectrum_constant_texture.h>

const SpectrumConstantTexture *
SpectrumConstantTexture::create(const ParameterDictionary &parameters,
                                const SpectrumType spectrum_type, GPUMemoryAllocator &allocator) {
    auto c = parameters.get_spectrum("value", spectrum_type, allocator);
    if (c == nullptr) {
        c = Spectrum::create_constant_spectrum(1.0, allocator);
    }

    auto texture = allocator.allocate<SpectrumConstantTexture>();
    texture->value = c;

    return texture;
}

PBRT_CPU_GPU
SampledSpectrum SpectrumConstantTexture::evaluate(const TextureEvalContext &ctx,
                                                  const SampledWavelengths &lambda) const {
    return value->sample(lambda);
}

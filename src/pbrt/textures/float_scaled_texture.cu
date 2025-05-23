#include <pbrt/base/float_texture.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/textures/float_scaled_texture.h>

FloatScaledTexture ::FloatScaledTexture(const ParameterDictionary &parameters,
                                        GPUMemoryAllocator &allocator) {
    scale = parameters.get_float_texture("scale", 1.0, allocator);
    texture = parameters.get_float_texture("tex", 1.0, allocator);
}

PBRT_CPU_GPU
Real FloatScaledTexture::evaluate(const TextureEvalContext &ctx) const {
    return scale->evaluate(ctx) * texture->evaluate(ctx);
}

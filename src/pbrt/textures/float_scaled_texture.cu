#include <pbrt/base/float_texture.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/textures/float_scaled_texture.h>
#include <pbrt/textures/texture_mapping_2d.h>

#include <pbrt/gpu/gpu_memory_allocator.h>

const FloatScaledTexture *FloatScaledTexture::create(const ParameterDictionary &parameters,
                                                     GPUMemoryAllocator &allocator) {
    auto float_scaled_texture = allocator.allocate<FloatScaledTexture>();

    float_scaled_texture->scale = parameters.get_float_texture("scale", 1.0, allocator);
    float_scaled_texture->texture = parameters.get_float_texture("tex", 1.0, allocator);

    return float_scaled_texture;
}

PBRT_CPU_GPU
Real FloatScaledTexture::evaluate(const TextureEvalContext &ctx) const {
    return scale->evaluate(ctx) * texture->evaluate(ctx);
}

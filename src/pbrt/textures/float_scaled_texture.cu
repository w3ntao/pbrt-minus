#include "pbrt/base/float_texture.h"
#include "pbrt/scene/parameter_dictionary.h"
#include "pbrt/textures/float_scaled_texture.h"
#include "pbrt/textures/texture_mapping_2d.h"

const FloatScaledTexture *FloatScaledTexture::create(const ParameterDictionary &parameters,
                                                     std::vector<void *> &gpu_dynamic_pointers) {

    FloatScaledTexture *float_scaled_texture;
    CHECK_CUDA_ERROR(cudaMallocManaged(&float_scaled_texture, sizeof(FloatScaledTexture)));
    gpu_dynamic_pointers.push_back(float_scaled_texture);

    float_scaled_texture->scale = parameters.get_float_texture("scale", 1.0, gpu_dynamic_pointers);
    float_scaled_texture->texture = parameters.get_float_texture("tex", 1.0, gpu_dynamic_pointers);

    return float_scaled_texture;
}

PBRT_CPU_GPU
FloatType FloatScaledTexture::evaluate(const TextureEvalContext &ctx) const {
    return scale->evaluate(ctx) * texture->evaluate(ctx);
}

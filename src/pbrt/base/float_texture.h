#pragma once

#include <cuda/std/variant>
#include <pbrt/textures/float_constant_texture.h>
#include <pbrt/textures/float_image_texture.h>
#include <pbrt/textures/float_scaled_texture.h>

namespace HIDDEN {
using FloatTextureVariants =
    cuda::std::variant<FloatConstantTexture, FloatImageTexture, FloatScaledTexture>;
}

class FloatTexture : public HIDDEN::FloatTextureVariants {
    using HIDDEN::FloatTextureVariants::FloatTextureVariants;

  public:
    static const FloatTexture *create(const std::string &texture_type,
                                      const Transform &render_from_object,
                                      const ParameterDictionary &parameters,
                                      GPUMemoryAllocator &allocator);

    static const FloatTexture *create_constant_float_texture(Real val,
                                                             GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Real evaluate(const TextureEvalContext &ctx) const {
        return cuda::std::visit([&](auto &x) { return x.evaluate(ctx); }, *this);
    }
};

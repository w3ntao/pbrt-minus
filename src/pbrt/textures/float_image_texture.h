#pragma once

#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/textures/image_texture_base.h>

class FloatImageTexture : ImageTextureBase {
  public:
    static const FloatImageTexture *create(const Transform &render_from_object,
                                           const ParameterDictionary &parameters,
                                           GPUMemoryAllocator &allocator) {
        auto float_image_texture = allocator.allocate<FloatImageTexture>();

        float_image_texture->init_image_texture_base(render_from_object, parameters, allocator);

        return float_image_texture;
    }

    PBRT_CPU_GPU
    FloatType evaluate(const TextureEvalContext &ctx) const {
        TexCoord2D c = texture_mapping->map(ctx);
        // Texture coordinates are (0,0) in the lower left corner, but
        // image coordinates are (0,0) in the upper left.

        c.st[1] = 1 - c.st[1];
        auto v = this->scale * mipmap->filter(c.st)[0];

        return invert ? std::max<FloatType>(0, 1 - v) : v;
    }
};

#pragma once

#include <pbrt/textures/image_texture_base.h>
#include <pbrt/textures/mipmap.h>
#include <pbrt/textures/texture_mapping_2d.h>

class FloatImageTexture : ImageTextureBase {
  public:
    FloatImageTexture(const Transform &render_from_object, const ParameterDictionary &parameters,
                      GPUMemoryAllocator &allocator)
        : ImageTextureBase(render_from_object, parameters, allocator) {}

    PBRT_CPU_GPU
    Real evaluate(const TextureEvalContext &ctx) const {
        TexCoord2D c = texture_mapping->map(ctx);
        // Texture coordinates are (0,0) in the lower left corner, but
        // image coordinates are (0,0) in the upper left.

        c.st[1] = 1 - c.st[1];
        auto v = this->scale * mipmap->filter(c.st)[0];

        return invert ? std::max<Real>(0, 1 - v) : v;
    }
};

#pragma once

#include <pbrt/gpu/macro.h>

class FloatConstantTexture;
class FloatImageTexture;
class FloatScaledTexture;

class GPUMemoryAllocator;
class ParameterDictionary;
class TextureEvalContext;
class Transform;

class FloatTexture {
  public:
    enum class Type {
        constant,
        image,
        scale,
    };

    static const FloatTexture *create(const std::string &texture_type,
                                      const Transform &render_from_object,
                                      const ParameterDictionary &parameters,
                                      GPUMemoryAllocator &allocator);

    static const FloatTexture *create_constant_float_texture(Real val,
                                                             GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Real evaluate(const TextureEvalContext &ctx) const;

  private:
    Type type;
    const void *ptr = nullptr;

    void init(const FloatConstantTexture *float_constant_texture);

    void init(const FloatImageTexture *float_image_texture);

    void init(const FloatScaledTexture *float_scaled_texture);
};

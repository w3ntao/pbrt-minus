#pragma once

#include <pbrt/gpu/macro.h>

class FloatConstantTexture;
class FloatImageTexture;
class FloatScaledTexture;

class GPUMemoryAllocator;
class ParameterDictionary;
class Transform;

struct TextureEvalContext;

class FloatTexture {
  public:
    enum class Type {
        constant,
        image,
        scale,
    };

    explicit FloatTexture(const FloatConstantTexture *texture_ptr)
        : type(Type::constant), ptr(texture_ptr) {}

    explicit FloatTexture(const FloatImageTexture *texture_ptr)
        : type(Type::image), ptr(texture_ptr) {}

    explicit FloatTexture(const FloatScaledTexture *texture_ptr)
        : type(Type::scale), ptr(texture_ptr) {}

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
};

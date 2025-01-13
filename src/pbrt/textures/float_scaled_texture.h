#pragma once

#include <pbrt/gpu/macro.h>

class FloatTexture;
class GPUMemoryAllocator;
class ParameterDictionary;
class TextureEvalContext;

class FloatScaledTexture {
  public:
    static const FloatScaledTexture *create(const ParameterDictionary &parameters,
                                            GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    FloatType evaluate(const TextureEvalContext &ctx) const;

  private:
    const FloatTexture *texture;
    const FloatTexture *scale;
};

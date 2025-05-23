#pragma once

#include <pbrt/gpu/macro.h>

class FloatTexture;
class GPUMemoryAllocator;
class ParameterDictionary;
struct TextureEvalContext;

class FloatScaledTexture {
  public:
    FloatScaledTexture(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Real evaluate(const TextureEvalContext &ctx) const;

  private:
    const FloatTexture *texture = nullptr;
    const FloatTexture *scale = nullptr;
};

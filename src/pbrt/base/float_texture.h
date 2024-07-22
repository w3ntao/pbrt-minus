#pragma once

#include <string>
#include <vector>

#include "pbrt/util/macro.h"

class FloatConstantTexture;
class FloatImageTexture;
class FloatScaledTexture;

class ParameterDictionary;
class TextureEvalContext;

class FloatTexture {
  public:
    enum class Type {
        constant,
        image,
        scale,
    };

    static const FloatTexture *create(const std::string &texture_type,
                                      const ParameterDictionary &parameters,
                                      std::vector<void *> &gpu_dynamic_pointers);

    static const FloatTexture *
    create_constant_float_texture(FloatType val, std::vector<void *> &gpu_dynamic_pointers);

    void init(const FloatConstantTexture *float_constant_texture);

    void init(const FloatImageTexture *float_image_texture);

    void init(const FloatScaledTexture *float_scaled_texture);

    PBRT_CPU_GPU
    FloatType evaluate(const TextureEvalContext &ctx) const;

  private:
    Type type;
    const void *ptr;
};

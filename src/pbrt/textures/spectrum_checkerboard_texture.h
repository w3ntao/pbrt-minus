#pragma once

#include <pbrt/gpu/macro.h>

enum class SpectrumType;
class GPUMemoryAllocator;
class ParameterDictionary;
class SampledSpectrum;
class SampledWavelengths;
class SpectrumTexture;
class Transform;
struct TextureEvalContext;
struct TextureMapping2D;
struct TextureMapping3D;

class SpectrumCheckerboardTexture {
  public:
    SpectrumCheckerboardTexture(const Transform &renderFromTexture, SpectrumType spectrumType,
                                const ParameterDictionary &parameters,
                                GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    const TextureMapping2D *map2D = nullptr;
    const TextureMapping3D *map3D = nullptr;

    const SpectrumTexture *tex0 = nullptr;
    const SpectrumTexture *tex1 = nullptr;

    void init(const TextureMapping2D *_map2D, const TextureMapping3D *_map3D,
              const SpectrumTexture *_tex1, const SpectrumTexture *_tex2) {
        map2D = _map2D;
        map3D = _map3D;
        tex0 = _tex1;
        tex1 = _tex2;
    }
};

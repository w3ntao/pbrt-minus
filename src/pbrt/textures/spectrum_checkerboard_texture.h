#pragma once

#include <pbrt/base/spectrum.h>
#include <pbrt/textures/texture_mapping_2d.h>
#include <pbrt/textures/texture_mapping_3d.h>

class ParameterDictionary;
class SpectrumTexture;
class Transform;

class SpectrumCheckerboardTexture {
  public:
    static const SpectrumCheckerboardTexture *create(const Transform &renderFromTexture,
                                                     SpectrumType spectrumType,
                                                     const ParameterDictionary &parameters,
                                                     GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    SampledSpectrum evaluate(const TextureEvalContext &ctx, const SampledWavelengths &lambda) const;

  private:
    const TextureMapping2D *map2D;
    const TextureMapping3D *map3D;

    const SpectrumTexture *tex0;
    const SpectrumTexture *tex1;

    void init(const TextureMapping2D *_map2D, const TextureMapping3D *_map3D,
              const SpectrumTexture *_tex1, const SpectrumTexture *_tex2) {
        map2D = _map2D;
        map3D = _map3D;
        tex0 = _tex1;
        tex1 = _tex2;
    }
};

#pragma once

#include <pbrt/medium/media_util.h>

class DenselySampledSpectrum;
class GPUMemoryAllocator;
class ParameterDictionary;

struct Medium {
    // currently we support only homogeneous medium

    const DenselySampledSpectrum *sigma_a = nullptr;
    const DenselySampledSpectrum *sigma_s = nullptr;

    HGPhaseFunction phase;

    Medium(const Spectrum *_sigma_a, const Spectrum *_sigma_s, Real sigma_scale, Real g,
           GPUMemoryAllocator &allocator);

    static const Medium *create(const ParameterDictionary &parameters,
                                GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    [[nodiscard]]
    SampledSpectrum sample_sigma_t(const SampledWavelengths &lambda) const;
};

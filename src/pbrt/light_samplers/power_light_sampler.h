#pragma once

#include <pbrt/gpu/macro.h>

class Distribution1D;
class GPUMemoryAllocator;
class HashMap;
class Light;
struct LightSampleContext;
struct SampledLight;

class PowerLightSampler {
  public:
    PowerLightSampler(const Light **_lights, int _light_num, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    const Light *get_light_by_idx(int idx) const {
        return lights[idx];
    }

    PBRT_CPU_GPU
    pbrt::optional<SampledLight> sample(Real u) const;

    PBRT_CPU_GPU
    pbrt::optional<SampledLight> sample(const LightSampleContext &ctx, Real u) const;

    PBRT_CPU_GPU
    Real pmf(const Light *light) const;

    PBRT_CPU_GPU
    Real pmf(const LightSampleContext &ctx, const Light *light) const {
        return pmf(light);
    }

    int light_num = 0;

  private:
    const Light **lights = nullptr;
    const Distribution1D *lights_power_distribution = nullptr;

    const HashMap *light_ptr_to_idx = nullptr;
};

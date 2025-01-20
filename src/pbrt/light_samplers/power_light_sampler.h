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
    static const PowerLightSampler *create(const Light **lights, const uint light_num,
                                           GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    const Light *get_light_by_idx(int idx) const {
        return lights[idx];
    }

    PBRT_CPU_GPU
    pbrt::optional<SampledLight> sample(FloatType u) const;

    PBRT_CPU_GPU
    pbrt::optional<SampledLight> sample(const LightSampleContext &ctx, FloatType u) const;

    PBRT_CPU_GPU
    FloatType pmf(const Light *light) const;

    PBRT_CPU_GPU
    FloatType pmf(const LightSampleContext &ctx, const Light *light) const {
        return pmf(light);
    }

    uint light_num;

  private:
    const Light **lights;
    const Distribution1D *lights_power_distribution;

    const HashMap *light_ptr_to_idx;
};

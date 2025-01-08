#pragma once

#include "pbrt/util/macro.h"
#include <vector>

class Distribution1D;
class HashMap;
class Light;
class LightSampleContext;
class SampledLight;

class PowerLightSampler {
  public:
    static const PowerLightSampler *create(const Light **lights, const uint light_num,
                                           std::vector<void *> &gpu_dynamic_pointers);

    PBRT_GPU
    const Light *get_light_by_idx(int idx) const {
        return lights[idx];
    }

    PBRT_GPU
    pbrt::optional<SampledLight> sample(FloatType u) const;

    PBRT_GPU
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

    // TODO: rewrite Distribution1D with AliasTable: Ray Tracing Gem 2: chapter 21
    const HashMap *light_ptr_to_idx;
};

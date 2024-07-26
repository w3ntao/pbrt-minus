#pragma once

#include <vector>

#include <cuda/std/optional>
#include "pbrt/util/macro.h"

class HashMap;
class Light;
class LightSampleContext;
class SampledLight;

class PowerLightSampler {
  public:
    static const PowerLightSampler *create(const Light **lights, const uint light_num,
                                           std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU
    cuda::std::optional<SampledLight> sample(FloatType u) const;

    PBRT_CPU_GPU
    cuda::std::optional<SampledLight> sample(const LightSampleContext &ctx, FloatType u) const;

    PBRT_CPU_GPU
    FloatType pmf(const Light *light) const;

    PBRT_CPU_GPU
    FloatType pmf(const LightSampleContext &ctx, const Light *light) const {
        return pmf(light);
    }

  private:
    const Light **lights;
    const FloatType *lights_power;

    HashMap *light_ptr_to_idx;

    uint light_num;
};

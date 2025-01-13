#pragma once

#include <pbrt/base/light.h>
#include <pbrt/gpu/gpu_memory_allocator.h>

class UniformLightSampler {
  public:
    static const UniformLightSampler *create(const Light **lights, const uint light_num,
                                             GPUMemoryAllocator &allocator) {

        auto uniform_light_sampler = allocator.allocate<UniformLightSampler>();

        uniform_light_sampler->lights = lights;
        uniform_light_sampler->light_num = light_num;

        return uniform_light_sampler;
    }

    PBRT_CPU_GPU
    pbrt::optional<SampledLight> sample(FloatType u) const {
        if (light_num == 0) {
            return {};
        }

        const auto num_in_float = FloatType(light_num);
        const uint light_idx = clamp<uint>(uint(u * num_in_float), 0, light_num - 1);
        return SampledLight{.light = lights[light_idx], .p = FloatType(1.0 / num_in_float)};
    }

    PBRT_CPU_GPU
    pbrt::optional<SampledLight> sample(const LightSampleContext &ctx, FloatType u) const {
        return sample(u);
    }

    PBRT_CPU_GPU
    FloatType pmf(const Light *light) const {
        if (light_num == 0) {
            return 0;
        }
        return 1.0 / FloatType(light_num);
    }

    PBRT_CPU_GPU
    FloatType pmf(const LightSampleContext &ctx, const Light *light) const {
        return pmf(light);
    }

  private:
    const Light **lights;
    uint light_num;
};

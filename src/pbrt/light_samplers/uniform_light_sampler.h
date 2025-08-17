#pragma once

#include <pbrt/base/light.h>

class UniformLightSampler {
  public:
    UniformLightSampler(const Light **_lights, const int _light_num)
        : lights(_lights), light_num(_light_num) {}

    PBRT_CPU_GPU
    pbrt::optional<SampledLight> sample(Real u) const {
        if (light_num == 0) {
            return {};
        }

        const auto num_in_float = Real(light_num);
        const int light_idx = clamp<int>(int(u * num_in_float), 0, light_num - 1);
        return SampledLight{.light = lights[light_idx], .pdf = Real(1.0 / num_in_float)};
    }

    PBRT_CPU_GPU
    pbrt::optional<SampledLight> sample(const LightSampleContext &ctx, Real u) const {
        return sample(u);
    }

    PBRT_CPU_GPU
    Real pmf(const Light *light) const {
        if (light_num == 0) {
            return 0;
        }
        return 1.0 / Real(light_num);
    }

    PBRT_CPU_GPU
    Real pmf(const LightSampleContext &ctx, const Light *light) const {
        return pmf(light);
    }

  private:
    const Light **lights = nullptr;
    int light_num = 0;
};

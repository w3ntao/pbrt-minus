#pragma once

#include <cuda/std/optional>
#include "pbrt/util/macro.h"
#include "pbrt/base/light.h"

class PowerLightSampler {
  public:
    static const PowerLightSampler *create(const Light **lights, const uint light_num,
                                           std::vector<void *> &gpu_dynamic_pointers) {
        PowerLightSampler *power_light_sampler;
        CHECK_CUDA_ERROR(cudaMallocManaged(&power_light_sampler, sizeof(PowerLightSampler)));
        gpu_dynamic_pointers.push_back(power_light_sampler);

        power_light_sampler->lights = lights;
        power_light_sampler->light_num = light_num;

        FloatType *lights_power;
        CHECK_CUDA_ERROR(cudaMallocManaged(&lights_power, sizeof(FloatType) * light_num));
        gpu_dynamic_pointers.push_back(lights_power);

        SampledWavelengths lambda = SampledWavelengths::sample_visible(0.5f);
        FloatType total_power = 0;
        for (uint idx = 0; idx < light_num; ++idx) {
            auto light = lights[idx];
            auto phi = light->phi(lambda).safe_div(lambda.pdf_as_sampled_spectrum());
            lights_power[idx] = phi.average();

            total_power += lights_power[idx];
        }

        if (total_power == 0.0) {
            for (uint idx = 0; idx < light_num; ++idx) {
                lights_power[idx] = 1.0 / total_power;
            }
        } else {
            for (uint idx = 0; idx < light_num; ++idx) {
                lights_power[idx] /= total_power;
            }
        }

        power_light_sampler->lights_power = lights_power;

        return power_light_sampler;
    }

    PBRT_CPU_GPU
    cuda::std::optional<SampledLight> sample(FloatType u) const {
        if (light_num == 0) {
            return {};
        }

        auto probability = u;
        for (uint idx = 0; idx < light_num; ++idx) {
            // TODO: rewrite this part with binary search
            if (probability <= lights_power[idx]) {
                return SampledLight{.light = lights[idx], .p = lights_power[idx]};
            }
            probability -= lights_power[idx];
        }

        return SampledLight{.light = lights[light_num - 1], .p = lights_power[light_num - 1]};
    }

    PBRT_CPU_GPU
    cuda::std::optional<SampledLight> sample(const LightSampleContext &ctx, FloatType u) const {
        return sample(u);
    }

    PBRT_CPU_GPU
    FloatType pmf(const Light *light) const {
        // TODO: rewrite this part?
        for (uint idx = 0; idx < light_num; ++idx) {
            if (lights[idx] == light) {
                return lights_power[idx];
            }
        }

        REPORT_FATAL_ERROR();
        return NAN;
    }

    PBRT_CPU_GPU
    FloatType pmf(const LightSampleContext &ctx, const Light *light) const {
        return pmf(light);
    }

  private:
    const Light **lights;
    const FloatType *lights_power;

    uint light_num;
};

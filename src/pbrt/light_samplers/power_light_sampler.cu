#include "pbrt/light_samplers/power_light_sampler.h"

#include "pbrt/util/hash_map.h"
#include "pbrt/base/light.h"

const PowerLightSampler *PowerLightSampler::create(const Light **lights, const uint light_num,
                                                   std::vector<void *> &gpu_dynamic_pointers) {
    PowerLightSampler *power_light_sampler;
    CHECK_CUDA_ERROR(cudaMallocManaged(&power_light_sampler, sizeof(PowerLightSampler)));
    gpu_dynamic_pointers.push_back(power_light_sampler);

    power_light_sampler->lights = nullptr;
    power_light_sampler->lights_power = nullptr;
    power_light_sampler->light_ptr_to_idx = nullptr;
    power_light_sampler->light_num = 0;

    std::vector<std::pair<uint, FloatType>> light_idx_power(light_num);

    SampledWavelengths lambda = SampledWavelengths::sample_visible(0.5f);
    FloatType total_power = 0;
    for (uint idx = 0; idx < light_num; ++idx) {
        auto light = lights[idx];
        auto phi = light->phi(lambda).safe_div(lambda.pdf_as_sampled_spectrum());
        auto power = phi.average();

        total_power += power;
        light_idx_power[idx] = {idx, power};
    }

    std::sort(light_idx_power.begin(), light_idx_power.end(),
              [](const auto &left, const auto &right) { return left.second > right.second; });

    if (total_power == 0.0) {
        for (uint idx = 0; idx < light_num; ++idx) {
            auto item = light_idx_power[idx];
            item.second = 1.0;
            light_idx_power[idx] = item;
        }
    } else {
        for (uint idx = 0; idx < light_num; ++idx) {
            auto item = light_idx_power[idx];
            item.second = item.second / total_power;
            light_idx_power[idx] = item;
        }
    }

    const Light **sorted_lights;
    FloatType *lights_power;
    CHECK_CUDA_ERROR(cudaMallocManaged(&lights_power, sizeof(FloatType) * light_num));
    CHECK_CUDA_ERROR(cudaMallocManaged(&sorted_lights, sizeof(Light *) * light_num));
    gpu_dynamic_pointers.push_back(lights_power);
    gpu_dynamic_pointers.push_back(sorted_lights);

    for (auto idx = 0; idx < light_num; ++idx) {
        sorted_lights[idx] = lights[light_idx_power[idx].first];
        lights_power[idx] = light_idx_power[idx].second;
    }

    if (DEBUGGING) {
        for (uint idx = 0; idx < light_num - 1; ++idx) {
            if (lights_power[idx] < lights_power[idx + 1]) {
                REPORT_FATAL_ERROR();
            }
        }
    }

    auto light_to_idx = HashMap::create(light_num, gpu_dynamic_pointers);

    for (auto idx = 0; idx < light_num; ++idx) {
        light_to_idx->insert((uintptr_t)sorted_lights[idx], idx);
    }

    power_light_sampler->light_num = light_num;
    power_light_sampler->lights = sorted_lights;
    power_light_sampler->lights_power = lights_power;
    power_light_sampler->light_ptr_to_idx = light_to_idx;

    return power_light_sampler;
}

PBRT_CPU_GPU
cuda::std::optional<SampledLight> PowerLightSampler::sample(FloatType u) const {
    if (light_num == 0) {
        return {};
    }

    auto probability = u;
    for (uint idx = 0; idx < light_num; ++idx) {
        // TODO: rewrite this part with binary search
        // TODO: implement AliasTable in PBRT-v4
        if (probability <= lights_power[idx]) {
            return SampledLight{.light = lights[idx], .p = lights_power[idx]};
        }
        probability -= lights_power[idx];
    }

    return SampledLight{.light = lights[light_num - 1], .p = lights_power[light_num - 1]};
}

PBRT_CPU_GPU
cuda::std::optional<SampledLight> PowerLightSampler::sample(const LightSampleContext &ctx,
                                                            FloatType u) const {
    return sample(u);
}

PBRT_CPU_GPU
FloatType PowerLightSampler::pmf(const Light *light) const {
    auto light_idx = light_ptr_to_idx->lookup((uintptr_t)light);
    return lights_power[light_idx];
}

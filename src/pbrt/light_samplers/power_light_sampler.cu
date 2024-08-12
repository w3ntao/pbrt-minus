#include "pbrt/light_samplers/power_light_sampler.h"

#include "pbrt/util/hash_map.h"
#include "pbrt/base/light.h"

const PowerLightSampler *PowerLightSampler::create(const Light **lights, const uint light_num,
                                                   std::vector<void *> &gpu_dynamic_pointers) {
    PowerLightSampler *power_light_sampler;
    CHECK_CUDA_ERROR(cudaMallocManaged(&power_light_sampler, sizeof(PowerLightSampler)));
    gpu_dynamic_pointers.push_back(power_light_sampler);

    power_light_sampler->lights = nullptr;
    power_light_sampler->lights_pmf = nullptr;
    power_light_sampler->lights_cdf = nullptr;
    power_light_sampler->light_ptr_to_idx = nullptr;
    power_light_sampler->light_num = light_num;

    if (light_num == 0) {
        return power_light_sampler;
    }

    std::vector<std::pair<uint, FloatType>> light_idx_pmf(light_num);

    SampledWavelengths lambda = SampledWavelengths::sample_visible(0.5f);
    FloatType total_power = 0;
    for (uint idx = 0; idx < light_num; ++idx) {
        auto light = lights[idx];
        auto phi = light->phi(lambda).safe_div(lambda.pdf_as_sampled_spectrum());
        auto power = phi.average();

        total_power += power;
        light_idx_pmf[idx] = {idx, power};
    }

    std::sort(light_idx_pmf.begin(), light_idx_pmf.end(),
              [](const auto &left, const auto &right) { return left.second > right.second; });

    if (total_power == 0.0) {
        for (uint idx = 0; idx < light_num; ++idx) {
            auto item = light_idx_pmf[idx];
            item.second = 1.0;
            light_idx_pmf[idx] = item;
        }
    } else {
        for (uint idx = 0; idx < light_num; ++idx) {
            auto item = light_idx_pmf[idx];
            item.second = item.second / total_power;
            light_idx_pmf[idx] = item;
        }
    }

    const Light **sorted_lights;
    FloatType *lights_pmf;
    FloatType *lights_cdf;
    CHECK_CUDA_ERROR(cudaMallocManaged(&lights_pmf, sizeof(FloatType) * light_num));
    CHECK_CUDA_ERROR(cudaMallocManaged(&lights_cdf, sizeof(FloatType) * light_num));
    CHECK_CUDA_ERROR(cudaMallocManaged(&sorted_lights, sizeof(Light *) * light_num));
    gpu_dynamic_pointers.push_back(lights_pmf);
    gpu_dynamic_pointers.push_back(sorted_lights);
    gpu_dynamic_pointers.push_back(lights_cdf);

    for (auto idx = 0; idx < light_num; ++idx) {
        sorted_lights[idx] = lights[light_idx_pmf[idx].first];
        lights_pmf[idx] = light_idx_pmf[idx].second;
    }

    if (DEBUGGING) {
        for (uint idx = 0; idx < light_num - 1; ++idx) {
            if (lights_pmf[idx] < lights_pmf[idx + 1]) {
                REPORT_FATAL_ERROR();
            }
        }
    }

    lights_cdf[0] = lights_pmf[0];
    for (size_t idx = 1; idx < light_num; ++idx) {
        lights_cdf[idx] = lights_cdf[idx - 1] + lights_pmf[idx];
    }

    auto light_to_idx = HashMap::create(light_num, gpu_dynamic_pointers);

    for (auto idx = 0; idx < light_num; ++idx) {
        light_to_idx->insert((uintptr_t)sorted_lights[idx], idx);
    }

    power_light_sampler->light_num = light_num;
    power_light_sampler->lights = sorted_lights;
    power_light_sampler->light_ptr_to_idx = light_to_idx;
    power_light_sampler->lights_pmf = lights_pmf;
    power_light_sampler->lights_cdf = lights_cdf;

    return power_light_sampler;
}

PBRT_CPU_GPU
cuda::std::optional<SampledLight> PowerLightSampler::sample(const FloatType u) const {
    if (light_num == 0) {
        return {};
    }

    if (light_num == 1) {
        return SampledLight{.light = lights[0], .p = lights_pmf[0]};
    }

    size_t light_idx;
    if (u < lights_cdf[0]) {
        light_idx = 0;
    } else {
        size_t start = 1;
        size_t end = light_num;

        while (true) {
            if (end - start <= 10) {
                light_idx = start;
                for (auto idx = start; idx < end; ++idx) {
                    if (u >= lights_cdf[idx - 1] && u < lights_cdf[idx]) {
                        light_idx = idx;
                        break;
                    }
                }
                break;
            } else {
                auto mid = (start + end) / 2;
                if (u >= lights_cdf[mid]) {
                    // notice: change pivot to mid+1, rather than mid
                    start = mid + 1;
                } else {
                    end = mid + 1;
                }
            }
        }
    }

    if (DEBUGGING) {
        if (light_idx < 0 || light_idx >= light_num) {
            REPORT_FATAL_ERROR();
        }
    }

    return SampledLight{.light = lights[light_idx], .p = lights_pmf[light_idx]};
}

PBRT_CPU_GPU
cuda::std::optional<SampledLight> PowerLightSampler::sample(const LightSampleContext &ctx,
                                                            FloatType u) const {
    return sample(u);
}

PBRT_CPU_GPU
FloatType PowerLightSampler::pmf(const Light *light) const {
    if (light_num == 0) {
        return 0;
    }

    auto light_idx = light_ptr_to_idx->lookup((uintptr_t)light);
    return lights_pmf[light_idx];
}

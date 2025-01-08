#include "pbrt/base/light.h"
#include "pbrt/light_samplers/power_light_sampler.h"
#include "pbrt/util/distribution_1d.h"
#include "pbrt/util/hash_map.h"

const PowerLightSampler *PowerLightSampler::create(const Light **lights, const uint light_num,
                                                   std::vector<void *> &gpu_dynamic_pointers) {
    PowerLightSampler *power_light_sampler;
    CHECK_CUDA_ERROR(cudaMallocManaged(&power_light_sampler, sizeof(PowerLightSampler)));
    gpu_dynamic_pointers.push_back(power_light_sampler);

    power_light_sampler->lights = nullptr;
    power_light_sampler->lights_power_distribution = nullptr;
    power_light_sampler->light_ptr_to_idx = nullptr;
    power_light_sampler->light_num = light_num;

    if (light_num == 0) {
        return power_light_sampler;
    }

    std::vector<FloatType> lights_pmf(light_num);
    SampledWavelengths lambda = SampledWavelengths::sample_visible(0.5f);
    for (uint idx = 0; idx < light_num; ++idx) {
        auto light = lights[idx];
        auto phi = light->phi(lambda).safe_div(lambda.pdf_as_sampled_spectrum());

        lights_pmf[idx] = phi.average();
    }

    Distribution1D *lights_power_distribution;
    CHECK_CUDA_ERROR(cudaMallocManaged(&lights_power_distribution, sizeof(Distribution1D)));
    gpu_dynamic_pointers.push_back(lights_power_distribution);

    lights_power_distribution->build(lights_pmf, gpu_dynamic_pointers);

    auto light_to_idx = HashMap::create(light_num, gpu_dynamic_pointers);

    for (auto idx = 0; idx < light_num; ++idx) {
        light_to_idx->insert((uintptr_t)lights[idx], idx);
    }

    power_light_sampler->light_num = light_num;
    power_light_sampler->lights = lights;
    power_light_sampler->light_ptr_to_idx = light_to_idx;
    power_light_sampler->lights_power_distribution = lights_power_distribution;

    return power_light_sampler;
}

PBRT_GPU
pbrt::optional<SampledLight> PowerLightSampler::sample(const FloatType u) const {
    if (light_num == 0) {
        return {};
    }

    auto result = lights_power_distribution->sample(u);

    if (DEBUG_MODE) {
        if (result.first >= light_num) {
            REPORT_FATAL_ERROR();
        }
    }

    return SampledLight{.light = lights[result.first], .p = result.second};
}

PBRT_GPU
pbrt::optional<SampledLight> PowerLightSampler::sample(const LightSampleContext &ctx,
                                                            FloatType u) const {
    return sample(u);
}

PBRT_CPU_GPU
FloatType PowerLightSampler::pmf(const Light *light) const {
    if (light_num == 0) {
        return 0;
    }

    auto light_idx = light_ptr_to_idx->lookup((uintptr_t)light);
    return lights_power_distribution->get_pdf(light_idx);
}

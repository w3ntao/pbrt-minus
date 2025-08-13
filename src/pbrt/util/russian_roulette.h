#pragma once

#include <pbrt/gpu/macro.h>

constexpr int depth_russian_roulette = 8;
constexpr Real russian_roulette_upper_bound = 0.95;

// depth-8 and clamped-to-0.95 are taken from Mitsuba

PBRT_CPU_GPU
static bool russian_roulette(SampledSpectrum &throughput, Sampler *sampler) {
    const auto survive_prob =
        std::fmin(throughput.max_component_value(), russian_roulette_upper_bound);
    if (sampler->get_1d() > survive_prob) {
        return true;
    }

    throughput /= survive_prob;

    return false;
}

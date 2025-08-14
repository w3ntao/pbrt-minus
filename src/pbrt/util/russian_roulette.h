#pragma once

#include <pbrt/gpu/macro.h>

constexpr int start_russian_roulette = 8;

namespace HIDDEN {
constexpr Real russian_roulette_upper_bound = 0.95;
}
// depth-8 and clamped-to-0.95 are taken from Mitsuba

PBRT_CPU_GPU
static bool russian_roulette(SampledSpectrum &throughput, Sampler *sampler,
                             int *next_time_russian_roulette) {
    if (next_time_russian_roulette) {
        *next_time_russian_roulette += 1;
        // to prevent russian roulette from triggering too often by material-less hit
    }

    const auto survive_prob =
        std::fmin(throughput.max_component_value(), HIDDEN::russian_roulette_upper_bound);
    if (sampler->get_1d() > survive_prob) {
        throughput = SampledSpectrum(0);
        return true;
    }

    throughput /= survive_prob;
    return false;
}

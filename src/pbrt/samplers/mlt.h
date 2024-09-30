#pragma once

#include "pbrt/util/rng.h"
#include "pbrt/util/stack.h"
#include <cmath>

struct PrimarySample {
    FloatType value;
    int modify_time;

    PBRT_CPU_GPU
    PrimarySample(FloatType _value, int _modify_time) : value(_value), modify_time(_modify_time) {}
};

const size_t PRIMARY_SAMPLE_SIZE = 64;

class MLTSampler {
  private:
    RNG rng;
    int sample_idx;

    PrimarySample samples[PRIMARY_SAMPLE_SIZE];
    Stack<PrimarySample, PRIMARY_SAMPLE_SIZE> backup_samples;

    PBRT_CPU_GPU
    inline FloatType mutate_small_step(const FloatType x) {
        // taken from PSSMLT
        // TODO: PBRT-v4's implements this mutation differently

        constexpr FloatType s1 = 1.0 / 512.0;
        constexpr FloatType s2 = 1.0 / 16.0;
        const FloatType r = rng.uniform<FloatType>();
        const FloatType dx = s1 / (s1 / s2 + std::abs(2.0 * r - 1.0)) - s1 / (s1 / s2 + 1.0);
        if (r < 0.5) {
            const FloatType x1 = x + dx;
            return (x1 < 1.0) ? x1 : x1 - 1.0;
        } else {
            const FloatType x1 = x - dx;
            return (x1 < 0.0) ? x1 + 1.0 : x1;
        }
    }

  public:
    int global_time;
    int large_step;
    int large_step_time;

    PBRT_CPU_GPU
    void init(int seed) {
        global_time = 0;
        large_step_time = 0;
        sample_idx = 0;
        large_step = 0;

        rng.set_sequence(seed, 0);

        for (uint idx = 0; idx < PRIMARY_SAMPLE_SIZE; ++idx) {
            samples[idx] = PrimarySample(rng.uniform<FloatType>(), 0);
        }
    }

    PBRT_CPU_GPU
    void init_sample_idx() {
        sample_idx = 0;
    }

    PBRT_CPU_GPU void clear_backup_samples() {
        backup_samples.clear();
    }

    PBRT_CPU_GPU
    void recover_samples() {
        for (int idx = sample_idx - 1; idx >= 0; --idx) {
            if (backup_samples.empty()) {
                return;
            }

            samples[idx] = backup_samples.pop();
        }
    }

    PBRT_CPU_GPU
    inline FloatType next_sample() {
        if (sample_idx < 0 || sample_idx >= PRIMARY_SAMPLE_SIZE) {
            REPORT_FATAL_ERROR();
        }

        if (samples[sample_idx].modify_time < global_time) {
            if (large_step > 0) {
                backup_samples.push(samples[sample_idx]);
                samples[sample_idx].modify_time = global_time;
                samples[sample_idx].value = rng.uniform<FloatType>();
            } else {
                // small step

                if (samples[sample_idx].modify_time < large_step_time) {
                    samples[sample_idx].modify_time = large_step_time;
                    samples[sample_idx].value = rng.uniform<FloatType>();
                }

                for (int idx = 0; idx < global_time - 1 - samples[sample_idx].modify_time; ++idx) {
                    samples[sample_idx].value = mutate_small_step(samples[sample_idx].value);
                }

                backup_samples.push(samples[sample_idx]);
                samples[sample_idx].value = mutate_small_step(samples[sample_idx].value);
                samples[sample_idx].modify_time = global_time;
            }
        }

        auto value = samples[sample_idx].value;
        sample_idx += 1;

        return value;
    }
};

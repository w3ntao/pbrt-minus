#pragma once

#include <pbrt/euclidean_space/point2.h>
#include <pbrt/spectrum_util/sampled_wavelengths.h>
#include <pbrt/util/rng.h>

struct PrimarySample {
    PBRT_CPU_GPU
    PrimarySample() : value(0), last_modification_iteration(0), value_backup(0), modify_backup(0) {}

    Real value;
    // PrimarySample Public Methods
    PBRT_CPU_GPU
    void backup() {
        value_backup = value;
        modify_backup = last_modification_iteration;
    }

    PBRT_CPU_GPU
    void restore() {
        value = value_backup;
        last_modification_iteration = modify_backup;
    }

    // PrimarySample Public Members
    int64_t last_modification_iteration;
    Real value_backup;
    int64_t modify_backup;
};

class MLTSampler {
    static constexpr size_t LENGTH = 256;

  public:
    RNG rng;

    PrimarySample samples[LENGTH];

    // MLTSampler Private Members
    int mutations_per_pixel;
    Real sigma;
    Real large_step_probability;

    int stream_count;
    int64_t current_iteration;
    bool large_step;
    int64_t last_large_step_iteration;
    int stream_index;
    int sample_index;

    PBRT_CPU_GPU
    void ensure_ready(int index);

    PBRT_CPU_GPU
    void setup_config(const int _mutation_per_pixel, const Real _sigma,
                      const Real _large_step_probability, const int _stream_count) {
        mutations_per_pixel = _mutation_per_pixel;
        sigma = _sigma;
        large_step_probability = _large_step_probability;

        stream_count = _stream_count;
        // for Path: stream_count = 1
        // for BDPT: stream_count = 3
    }

    PBRT_CPU_GPU
    void init(const long rng_sequence_index) {
        rng.set_sequence(MixBits(rng_sequence_index));

        current_iteration = 0;
        last_large_step_iteration = 0;
        large_step = true;
    }

    PBRT_CPU_GPU
    void advance(int64_t idelta) {
        rng.advance(idelta);
    }

    PBRT_CPU_GPU
    void start_iteration();

    PBRT_CPU_GPU
    void accept();

    PBRT_CPU_GPU
    void reject();

    PBRT_CPU_GPU
    void start_stream(int index);

    PBRT_CPU_GPU
    int get_next_index() {
        return stream_index + stream_count * sample_index++;
    }

    PBRT_CPU_GPU
    int get_samples_per_pixel() const {
        return mutations_per_pixel;
    }

    PBRT_CPU_GPU
    Real get_1d();

    PBRT_CPU_GPU
    Point2f get_2d();

    PBRT_CPU_GPU
    Point2f get_pixel_2d();
};

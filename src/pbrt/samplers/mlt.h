#pragma once

#include <pbrt/euclidean_space/point2.h>
#include <pbrt/spectrum_util/sampled_wavelengths.h>
#include <pbrt/util/rng.h>
#include <vector>

struct PrimarySample {
    PrimarySample() : value(0), last_modification_iteration(0), value_backup(0), modify_backup(0) {}

    Real value;
    // PrimarySample Public Methods
    void backup() {
        value_backup = value;
        modify_backup = last_modification_iteration;
    }

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
  public:
    ~MLTSampler() {
        if (DEBUG_MODE) {
            auto max_samples = stream_index + stream_count * sample_index;
            printf("    %s(): max samples used: %d\n", __FUNCTION__, max_samples);
        }
    }

    static Real compute_importance(int depth, int max_depth, Real luminance) {
        // Keep a nonzero importance for depth-zero paths that directly hit an area light.
        return (depth + 0.001) / max_depth * std::sqrt(luminance);
    }

    RNG rng;

    std::vector<PrimarySample> samples;

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

    void ensure_ready(int index);

    void setup_config(const int _mutation_per_pixel, const Real _sigma, const Real _large_step_probability,
                      const int _stream_count) {
        mutations_per_pixel = _mutation_per_pixel;
        sigma = _sigma;
        large_step_probability = _large_step_probability;

        stream_count = _stream_count;
        // for Path: stream_count = 1
        // for BDPT: stream_count = 3
    }

    void init(const long rng_sequence_index) {
        rng.set_sequence(MixBits(rng_sequence_index));

        current_iteration = 0;
        last_large_step_iteration = 0;
        large_step = true;
    }

    void advance(int64_t idelta) {
        rng.advance(idelta);
    }

    void start_iteration();

    void accept();

    void reject();

    void start_stream(int index);

    int get_next_index() {
        return stream_index + stream_count * sample_index++;
    }

    int get_samples_per_pixel() const {
        return mutations_per_pixel;
    }

    Real get_1d();

    Point2f get_2d();

    Point2f get_pixel_2d();
};

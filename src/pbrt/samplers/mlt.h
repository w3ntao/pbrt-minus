#pragma once

#include "pbrt/euclidean_space/point2.h"
#include "pbrt/util/hash.h"
#include "pbrt/util/rng.h"
#include <cmath>

constexpr FloatType global_sigma = 0.01;
constexpr FloatType global_largeStepProbability = 0.3;

struct PrimarySample {
    PBRT_CPU_GPU
    PrimarySample() : value(0), valueBackup(0), modifyBackup(0), lastModificationIteration(0) {}

    FloatType value = 0;
    // PrimarySample Public Methods
    PBRT_CPU_GPU
    void Backup() {
        valueBackup = value;
        modifyBackup = lastModificationIteration;
    }

    PBRT_CPU_GPU
    void Restore() {
        value = valueBackup;
        lastModificationIteration = modifyBackup;
    }

    // PrimarySample Public Members
    int64_t lastModificationIteration = 0;
    FloatType valueBackup = 0;
    int64_t modifyBackup = 0;
};

class MLTSampler {
    static const size_t LENGTH = 64;

  public:
    RNG rng;

    PrimarySample X[LENGTH];

    // MLTSampler Private Members
    int mutationsPerPixel;
    FloatType sigma;
    FloatType largeStepProbability;

    int streamCount;
    int64_t currentIteration = 0;
    bool largeStep = true;
    int64_t lastLargeStepIteration = 0;
    int streamIndex;
    int sampleIndex;

    PBRT_CPU_GPU
    void EnsureReady(int index);

    PBRT_CPU_GPU
    void init(long rngSequenceIndex) {
        rng.set_sequence(MixBits(rngSequenceIndex));

        currentIteration = 0;
        lastLargeStepIteration = 0;
        largeStep = true;

        mutationsPerPixel = 4;
        sigma = global_sigma;
        largeStepProbability = global_largeStepProbability;
        streamCount = 1;
        // TODO: change streamCount
    }

    PBRT_CPU_GPU
    void StartIteration();

    PBRT_CPU_GPU
    void Accept();

    PBRT_CPU_GPU
    void Reject();

    PBRT_CPU_GPU
    void StartStream(int index);

    PBRT_CPU_GPU
    int GetNextIndex() {
        return streamIndex + streamCount * sampleIndex++;
    }

    PBRT_CPU_GPU
    int SamplesPerPixel() const {
        return mutationsPerPixel;
    }

    PBRT_CPU_GPU
    void StartPixelSample(uint pixel_idx, const int sampleIndex, const int dim) {
        rng.set_sequence(pbrt::hash(pixel_idx));
        rng.advance(sampleIndex * 65536 + dim * 8192);
    }

    PBRT_CPU_GPU
    FloatType Get1D();

    PBRT_CPU_GPU
    Point2f Get2D();

    PBRT_CPU_GPU
    Point2f GetPixel2D();
};

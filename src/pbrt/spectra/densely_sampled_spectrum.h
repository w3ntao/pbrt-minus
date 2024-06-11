#pragma once

#include <vector>

#include "pbrt/spectrum_util/spectrum_constants_cie.h"
#include "pbrt/spectrum_util/black_body_spectrum.h"

class Spectrum;

class DenselySampledSpectrum {
  public:
    PBRT_CPU_GPU
    FloatType inner_product(const Spectrum *spectrum) const;

    PBRT_CPU_GPU
    void init_from_spectrum(const Spectrum *spectrum);

    template <typename F>
    PBRT_CPU_GPU void init_with_sample_function(F func, uint lambda_min = LAMBDA_MIN,
                                                uint lambda_max = LAMBDA_MAX) {
        for (uint lambda = lambda_min; lambda <= lambda_max; ++lambda) {
            values[lambda - lambda_min] = func(lambda);
        }
    }

    PBRT_CPU_GPU
    bool operator==(const DenselySampledSpectrum &_spectrum) const {
        for (uint i = 0; i < LAMBDA_RANGE; ++i) {
            if (values[i] != _spectrum.values[i]) {
                return false;
            }
        }

        return true;
    }

    PBRT_CPU_GPU
    FloatType operator()(FloatType lambda) const {
        const int floor = std::floor(lambda);
        const int ceil = std::ceil(lambda);

        if (floor < LAMBDA_MIN || ceil > LAMBDA_MAX) {
            return 0.0;
        }

        return lerp(lambda - floor, values[floor - LAMBDA_MIN], values[ceil - LAMBDA_MIN]);
    }

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const {
        SampledSpectrum sampled_values;

        for (uint i = 0; i < NSpectrumSamples; ++i) {
            int floor = std::floor(lambda[i]);
            int ceil = std::ceil(lambda[i]);
            if (floor < LAMBDA_MIN || ceil > LAMBDA_MAX) {
                sampled_values[i] = 0;
            } else {
                sampled_values[i] =
                    lerp(lambda[i] - floor, values[floor - LAMBDA_MIN], values[ceil - LAMBDA_MIN]);
            }
        }

        return sampled_values;
    }

    PBRT_CPU_GPU
    void scale(FloatType s) {
        for (uint i = 0; i < LAMBDA_RANGE; ++i) {
            values[i] = values[i] * s;
        }
    }

  private:
    FloatType values[LAMBDA_RANGE];
};

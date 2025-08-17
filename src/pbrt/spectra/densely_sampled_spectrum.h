#pragma once

#include <pbrt/spectrum_util/sampled_wavelengths.h>
#include <pbrt/spectrum_util/spectrum_constants_cie.h>

class DenselySampledSpectrum {
  public:
    DenselySampledSpectrum() {}

    DenselySampledSpectrum(const Spectrum *spectrum, const Real scale) {
        for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
            values[lambda - LAMBDA_MIN] = (*spectrum)(lambda)*scale;
        }
    }

    PBRT_CPU_GPU
    Real inner_product(const Spectrum *spectrum) const {
        Real sum = 0;
        for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
            sum += (*this)(lambda) * (*spectrum)(lambda);
        }

        return sum;
    }

    template <typename F>
    PBRT_CPU_GPU void init_with_sample_function(F func, int lambda_min = LAMBDA_MIN,
                                                int lambda_max = LAMBDA_MAX) {
        for (int lambda = lambda_min; lambda <= lambda_max; ++lambda) {
            values[lambda - lambda_min] = func(lambda);
        }
    }

    PBRT_CPU_GPU
    bool operator==(const DenselySampledSpectrum &_spectrum) const {
        for (int i = 0; i < LAMBDA_RANGE; ++i) {
            if (values[i] != _spectrum.values[i]) {
                return false;
            }
        }

        return true;
    }

    PBRT_CPU_GPU
    Real operator()(Real lambda) const {
        const int floor = std::floor(lambda);
        const int ceil = std::ceil(lambda);

        if (floor < LAMBDA_MIN || ceil > LAMBDA_MAX) {
            return 0.0;
        }

        return pbrt::lerp(lambda - floor, values[floor - LAMBDA_MIN], values[ceil - LAMBDA_MIN]);
    }

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const {
        SampledSpectrum sampled_values;

        for (int i = 0; i < NSpectrumSamples; ++i) {
            int floor = std::floor(lambda[i]);
            int ceil = std::ceil(lambda[i]);
            if (floor < LAMBDA_MIN || ceil > LAMBDA_MAX) {
                sampled_values[i] = 0;
            } else {
                sampled_values[i] = pbrt::lerp(lambda[i] - floor, values[floor - LAMBDA_MIN],
                                               values[ceil - LAMBDA_MIN]);
            }
        }

        return sampled_values;
    }

  private:
    Real values[LAMBDA_RANGE]{NAN};
};

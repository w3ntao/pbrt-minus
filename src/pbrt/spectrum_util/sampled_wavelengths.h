#pragma once

#include "pbrt/spectrum_util/constants.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"
#include "pbrt/util/utility_math.h"
#include "pbrt/util/sampling.h"

// SampledWavelengths Definitions
class SampledWavelengths {
  public:
    PBRT_CPU_GPU SampledWavelengths() {
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            lambda[i] = 0.0;
            pdf[i] = 0.0;
        }
    }

    PBRT_CPU_GPU SampledWavelengths(const FloatType _lambda[NSpectrumSamples],
                                    const FloatType _pdf[NSpectrumSamples]) {
        for (uint idx = 0; idx < NSpectrumSamples; ++idx) {
            lambda[idx] = _lambda[idx];
            pdf[idx] = _pdf[idx];
        }
    }

    PBRT_CPU_GPU
    static SampledWavelengths SampleUniform(FloatType u, FloatType lambda_min = LAMBDA_MIN,
                                            FloatType lambda_max = LAMBDA_MAX) {
        FloatType _lambda[NSpectrumSamples];

        // Sample first wavelength using _u_
        _lambda[0] = lerp(u, lambda_min, lambda_max);

        // Initialize _lambda_ for remaining wavelengths
        FloatType delta = (lambda_max - lambda_min) / NSpectrumSamples;
        for (uint i = 1; i < NSpectrumSamples; ++i) {
            _lambda[i] = _lambda[i - 1] + delta;
            if (_lambda[i] > lambda_max) {
                _lambda[i] = lambda_min + (_lambda[i] - lambda_max);
            }
        }

        FloatType _pdf[NSpectrumSamples];
        // Compute PDF for sampled wavelengths
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            _pdf[i] = 1 / (lambda_max - lambda_min);
        }

        return SampledWavelengths(_lambda, _pdf);
    }

    PBRT_CPU_GPU
    static SampledWavelengths sample_visible(FloatType _u) {
        FloatType _lambda[NSpectrumSamples];
        FloatType _pdf[NSpectrumSamples];

        for (uint i = 0; i < NSpectrumSamples; ++i) {
            // Compute _up_ for $i$th wavelength sample
            FloatType u_prime = _u + FloatType(i) / NSpectrumSamples;
            if (u_prime > 1) {
                u_prime -= 1;
            }

            _lambda[i] = sample_visible_wavelengths(u_prime);
            _pdf[i] = visible_wavelengths_pdf(_lambda[i]);
        }

        return {_lambda, _pdf};
    }

    PBRT_CPU_GPU
    bool operator==(const SampledWavelengths &swl) const {
        for (uint idx = 0; idx < NSpectrumSamples; ++idx) {
            if (lambda[idx] != swl.lambda[idx]) {
                return false;
            }

            if (pdf[idx] != swl.pdf[idx]) {
                return false;
            }
        }

        return true;
    }
    PBRT_CPU_GPU
    bool operator!=(const SampledWavelengths &swl) const {
        return !(*this == swl);
    }

    PBRT_CPU_GPU
    FloatType operator[](uint i) const {
        return lambda[i];
    }

    PBRT_CPU_GPU
    FloatType &operator[](uint i) {
        return lambda[i];
    }

    PBRT_CPU_GPU
    SampledSpectrum pdf_as_sampled_spectrum() const {
        FloatType _pdf[NSpectrumSamples];
        for (uint idx = 0; idx < NSpectrumSamples; ++idx) {
            _pdf[idx] = pdf[idx];
        }

        return SampledSpectrum(_pdf);
    }

    PBRT_CPU_GPU
    void TerminateSecondary() {
        if (SecondaryTerminated()) {
            return;
        }

        // Update wavelength probabilities for termination
        for (uint i = 1; i < NSpectrumSamples; ++i) {
            pdf[i] = 0;
        }

        pdf[0] /= NSpectrumSamples;
    }

    PBRT_CPU_GPU
    bool SecondaryTerminated() const {
        for (uint i = 1; i < NSpectrumSamples; ++i) {
            if (pdf[i] != 0) {
                return false;
            }
        }
        return true;
    }

    PBRT_CPU_GPU void print() const {
        printf("lambda: [");
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            printf("%f, ", lambda[i]);
        }
        printf("]\n");
        printf("pdf: [");
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            printf("%f, ", pdf[i]);
        }
        printf("]\n");
    }

  private:
    FloatType lambda[NSpectrumSamples];
    FloatType pdf[NSpectrumSamples];
};

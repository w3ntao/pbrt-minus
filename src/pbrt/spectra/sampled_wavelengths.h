#pragma once

#include "pbrt/util/utility_math.h"
#include "pbrt/util/sampling.h"
#include "pbrt/spectra/constants.h"
#include "pbrt/spectra/sampled_spectrum.h"

// SampledWavelengths Definitions
class SampledWavelengths {
  public:
    PBRT_CPU_GPU SampledWavelengths() {
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            lambda[i] = 0.0;
            pdf[i] = 0.0;
        }
    }

    PBRT_CPU_GPU SampledWavelengths(const double _lambda[NSpectrumSamples],
                                    const double _pdf[NSpectrumSamples]) {
        for (uint idx = 0; idx < NSpectrumSamples; ++idx) {
            lambda[idx] = _lambda[idx];
            pdf[idx] = _pdf[idx];
        }
    }

    PBRT_CPU_GPU
    static SampledWavelengths SampleUniform(double u, double lambda_min = LAMBDA_MIN,
                                            double lambda_max = LAMBDA_MAX) {
        double _lambda[NSpectrumSamples];

        // Sample first wavelength using _u_
        _lambda[0] = lerp(u, lambda_min, lambda_max);

        // Initialize _lambda_ for remaining wavelengths
        double delta = (lambda_max - lambda_min) / NSpectrumSamples;
        for (uint i = 1; i < NSpectrumSamples; ++i) {
            _lambda[i] = _lambda[i - 1] + delta;
            if (_lambda[i] > lambda_max) {
                _lambda[i] = lambda_min + (_lambda[i] - lambda_max);
            }
        }

        double _pdf[NSpectrumSamples];
        // Compute PDF for sampled wavelengths
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            _pdf[i] = 1 / (lambda_max - lambda_min);
        }

        return SampledWavelengths(_lambda, _pdf);
    }

    PBRT_CPU_GPU
    static SampledWavelengths sample_visible(double _u) {
        double _lambda[NSpectrumSamples];
        double _pdf[NSpectrumSamples];

        for (uint i = 0; i < NSpectrumSamples; ++i) {
            // Compute _up_ for $i$th wavelength sample
            double u_prime = _u + double(i) / NSpectrumSamples;
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
    double operator[](uint i) const {
        return lambda[i];
    }

    PBRT_CPU_GPU
    double &operator[](uint i) {
        return lambda[i];
    }

    PBRT_CPU_GPU
    SampledSpectrum pdf_as_sampled_spectrum() const {
        double _pdf[NSpectrumSamples];
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
    double lambda[NSpectrumSamples];
    double pdf[NSpectrumSamples];
};

#pragma once

#include <array>

#include "pbrt/util/math.h"
#include "pbrt/util/sampling.h"
#include "pbrt/spectra/constants.h"
#include "pbrt/spectra/sampled_spectrum.h"

// SampledWavelengths Definitions
class SampledWavelengths {
  public:
    PBRT_CPU_GPU SampledWavelengths() {
        for (int i = 0; i < NSpectrumSamples; ++i) {
            lambda[i] = 0.0;
            pdf[i] = 0.0;
        }
    }

    PBRT_CPU_GPU SampledWavelengths(const std::array<double, NSpectrumSamples> &_lambda,
                                    const std::array<double, NSpectrumSamples> &_pdf)
        : lambda(_lambda), pdf(_pdf) {}

    PBRT_CPU_GPU
    static SampledWavelengths SampleUniform(double u, double lambda_min = LAMBDA_MIN,
                                            double lambda_max = LAMBDA_MAX) {
        std::array<double, NSpectrumSamples> lambda;

        // Sample first wavelength using _u_
        lambda[0] = lerp(u, lambda_min, lambda_max);

        // Initialize _lambda_ for remaining wavelengths
        double delta = (lambda_max - lambda_min) / NSpectrumSamples;
        for (int i = 1; i < NSpectrumSamples; ++i) {
            lambda[i] = lambda[i - 1] + delta;
            if (lambda[i] > lambda_max) {
                lambda[i] = lambda_min + (lambda[i] - lambda_max);
            }
        }

        std::array<double, NSpectrumSamples> pdf;
        // Compute PDF for sampled wavelengths
        for (int i = 0; i < NSpectrumSamples; ++i) {
            pdf[i] = 1 / (lambda_max - lambda_min);
        }

        return SampledWavelengths(lambda, pdf);
    }

    PBRT_CPU_GPU
    static SampledWavelengths sample_visible(double _u) {
        std::array<double, NSpectrumSamples> lambda;
        std::array<double, NSpectrumSamples> pdf;

        for (int i = 0; i < NSpectrumSamples; ++i) {
            // Compute _up_ for $i$th wavelength sample
            double u_prime = _u + double(i) / NSpectrumSamples;
            if (u_prime > 1) {
                u_prime -= 1;
            }

            lambda[i] = sample_visible_wavelengths(u_prime);
            pdf[i] = visible_wavelengths_pdf(lambda[i]);
        }

        return {lambda, pdf};
    }

    PBRT_CPU_GPU
    bool operator==(const SampledWavelengths &swl) const {
        return lambda == swl.lambda && pdf == swl.pdf;
    }
    PBRT_CPU_GPU
    bool operator!=(const SampledWavelengths &swl) const {
        return lambda != swl.lambda || pdf != swl.pdf;
    }

    PBRT_CPU_GPU
    double operator[](int i) const {
        return lambda[i];
    }

    PBRT_CPU_GPU
    double &operator[](int i) {
        return lambda[i];
    }

    PBRT_CPU_GPU
    SampledSpectrum pdf_as_sampled_spectrum() const {
        return SampledSpectrum(pdf);
    }

    PBRT_CPU_GPU
    void TerminateSecondary() {
        if (SecondaryTerminated()) {
            return;
        }

        // Update wavelength probabilities for termination
        for (int i = 1; i < NSpectrumSamples; ++i) {
            pdf[i] = 0;
        }

        pdf[0] /= NSpectrumSamples;
    }

    PBRT_CPU_GPU
    bool SecondaryTerminated() const {
        for (int i = 1; i < NSpectrumSamples; ++i) {
            if (pdf[i] != 0) {
                return false;
            }
        }
        return true;
    }

    PBRT_CPU_GPU void print() const {
        printf("lambda: [");
        for (int i = 0; i < NSpectrumSamples; ++i) {
            printf("%f, ", lambda[i]);
        }
        printf("]\n");
        printf("pdf: [");
        for (int i = 0; i < NSpectrumSamples; ++i) {
            printf("%f, ", pdf[i]);
        }
        printf("]\n");
    }

    // private:
    std::array<double, NSpectrumSamples> lambda;
    std::array<double, NSpectrumSamples> pdf;
};

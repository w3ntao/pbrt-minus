#pragma once

#include <array>

#include "pbrt/util/math.h"
#include "pbrt/spectra/constants.h"
#include "pbrt/spectra/sampled_spectrum.h"

// SampledWavelengths Definitions
class SampledWavelengths {
  public:
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

    // TODO: progress 2024/02/03: blocked by SampleVisibleWavelengths and VisibleWavelengthsPDF
    /*
    PBRT_CPU_GPU
    static SampledWavelengths SampleVisible(double u) {
        SampledWavelengths swl;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            // Compute _up_ for $i$th wavelength sample
            double up = u + double(i) / NSpectrumSamples;
            if (up > 1) {
                up -= 1;
            }

            swl.lambda[i] = SampleVisibleWavelengths(up);
            swl.pdf[i] = VisibleWavelengthsPDF(swl.lambda[i]);
        }
        return swl;
    }
    */

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

  private:
    std::array<double, NSpectrumSamples> lambda;
    std::array<double, NSpectrumSamples> pdf;
};

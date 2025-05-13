#pragma once

#include <pbrt/spectrum_util/sampled_spectrum.h>
#include <pbrt/spectrum_util/spectrum_constants_cie.h>
#include <pbrt/util/sampling.h>

class SampledWavelengths {
  public:
    PBRT_CPU_GPU
    SampledWavelengths() {
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            lambda[i] = 0.0;
            pdf[i] = 0.0;
        }
    }

    PBRT_CPU_GPU
    SampledWavelengths(const Real _lambda[NSpectrumSamples],
                       const Real _pdf[NSpectrumSamples]) {
        for (uint idx = 0; idx < NSpectrumSamples; ++idx) {
            lambda[idx] = _lambda[idx];
            pdf[idx] = _pdf[idx];
        }
    }

    PBRT_CPU_GPU
    static SampledWavelengths nan() {
        SampledWavelengths wavelengths;

        for (uint i = 0; i < NSpectrumSamples; ++i) {
            wavelengths.lambda[i] = NAN;
            wavelengths.pdf[i] = NAN;
        }

        return wavelengths;
    }

    PBRT_CPU_GPU
    static SampledWavelengths zero() {
        SampledWavelengths wavelengths;

        for (uint i = 0; i < NSpectrumSamples; ++i) {
            wavelengths.lambda[i] = 0;
            wavelengths.pdf[i] = 0;
        }

        return wavelengths;
    }

    PBRT_CPU_GPU
    bool has_nan() const {
        for (uint idx = 0; idx < NSpectrumSamples; ++idx) {
            if (std::isnan(lambda[idx]) || std::isinf(lambda[idx]) || std::isnan(pdf[idx]) ||
                std::isinf(pdf[idx])) {
                return true;
            }
        }

        return false;
    }

    PBRT_CPU_GPU
    static SampledWavelengths sample_visible(Real _u) {
        Real _lambda[NSpectrumSamples];
        Real _pdf[NSpectrumSamples];

        for (uint i = 0; i < NSpectrumSamples; ++i) {
            // Compute _up_ for $i$th wavelength sample
            Real u_prime = _u + Real(i) / NSpectrumSamples;
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
    Real operator[](uint i) const {
        return lambda[i];
    }

    PBRT_CPU_GPU
    Real &operator[](uint i) {
        return lambda[i];
    }

    PBRT_CPU_GPU
    SampledSpectrum pdf_as_sampled_spectrum() const {
        SampledSpectrum result;
        for (uint idx = 0; idx < NSpectrumSamples; ++idx) {
            result[idx] = pdf[idx];
        }

        return result;
    }

    PBRT_CPU_GPU
    void terminate_secondary() {
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

    PBRT_CPU_GPU
    void print() const {
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
    Real lambda[NSpectrumSamples];
    Real pdf[NSpectrumSamples];
};

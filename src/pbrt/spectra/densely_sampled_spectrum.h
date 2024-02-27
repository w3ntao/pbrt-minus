#pragma once

#include "pbrt/base/spectrum.h"
#include "pbrt/spectra/constants.h"

class DenselySampledSpectrum : public Spectrum {
  public:
    PBRT_GPU DenselySampledSpectrum() {
        for (int i = 0; i < values.size(); ++i) {
            values[i] = 0.0;
        }
    }

    PBRT_GPU
    explicit DenselySampledSpectrum(const Spectrum &_spectrum) {
        for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
            values[lambda - LAMBDA_MIN] = _spectrum(lambda);
        }
    }

    PBRT_GPU
    DenselySampledSpectrum(const DenselySampledSpectrum &_spectrum) : values(_spectrum.values) {}

    PBRT_GPU
    bool operator==(const DenselySampledSpectrum &_spectrum) const {
        for (int i = 0; i < values.size(); ++i) {
            if (values[i] != _spectrum.values[i]) {
                return false;
            }
        }

        return true;
    }

    PBRT_GPU
    double operator()(double lambda) const override {
        const int floor = std::floor(lambda);
        const int ceil = std::ceil(lambda);

        if (floor < LAMBDA_MIN || ceil > LAMBDA_MAX) {
            return 0.0;
        }

        return lerp(lambda - floor, values[floor - LAMBDA_MIN], values[ceil - LAMBDA_MIN]);
    }

    PBRT_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const override {
        std::array<double, NSpectrumSamples> sampled_values;

        for (int i = 0; i < NSpectrumSamples; ++i) {
            int floor = std::floor(lambda[i]);
            int ceil = std::ceil(lambda[i]);
            if (floor < LAMBDA_MIN || ceil > LAMBDA_MAX) {
                sampled_values[i] = 0;
            } else {
                sampled_values[i] =
                    lerp(lambda[i] - floor, values[floor - LAMBDA_MIN], values[ceil - LAMBDA_MIN]);
            }
        }

        return SampledSpectrum(sampled_values);
    }

    PBRT_GPU
    void scale(double s) {
        for (double &v : values) {
            v *= s;
        }
    }

  private:
    std::array<double, LAMBDA_RANGE> values;
};

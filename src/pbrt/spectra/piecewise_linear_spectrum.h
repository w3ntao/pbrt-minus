#pragma once

#include "pbrt/util/macro.h"
#include "pbrt/base/spectrum.h"

class PiecewiseLinearSpectrum : public Spectrum {
  public:
    PBRT_GPU PiecewiseLinearSpectrum() : length(0), lambdas(nullptr), values(nullptr) {}

    PBRT_GPU PiecewiseLinearSpectrum(const double *_lambdas, const double *_values, int _length)
        : length(_length) {
        auto temp_lambdas = new double[_length];
        auto temp_values = new double[_length];

        memcpy(temp_lambdas, _lambdas, sizeof(double) * length);
        memcpy(temp_values, _values, sizeof(double) * length);

        lambdas = temp_lambdas;
        values = temp_values;
    }

    PBRT_GPU PiecewiseLinearSpectrum(const PiecewiseLinearSpectrum &spectrum) {
        length = spectrum.length;
        auto temp_lambdas = new double[length];
        auto temp_values = new double[length];

        memcpy(temp_lambdas, spectrum.lambdas, sizeof(double) * length);
        memcpy(temp_values, spectrum.values, sizeof(double) * length);

        lambdas = temp_lambdas;
        values = temp_values;
    }

    PBRT_GPU ~PiecewiseLinearSpectrum() override {
        delete lambdas;
        delete values;
    }

    PBRT_GPU static PiecewiseLinearSpectrum from_interleaved(const double *samples, int num_samples,
                                                             bool normalize,
                                                             const Spectrum *cie_y) {
        const double _lambda_min = samples[0];
        const double _lambda_max = samples[num_samples - 2];

        double *lambdas = nullptr;
        double *values = nullptr;
        int n = -1;

        // Extend samples to cover range of visible wavelengths if needed.
        if (_lambda_min > LAMBDA_MIN && _lambda_max < LAMBDA_MAX) {
            n = num_samples / 2 + 2;
            lambdas = new double[n];
            values = new double[n];

            lambdas[0] = LAMBDA_MIN - 1;
            values[0] = samples[1];
            for (int i = 0; i < num_samples / 2; ++i) {
                lambdas[i + 1] = samples[i * 2];
                values[i + 1] = samples[i * 2 + 1];
            }
            lambdas[n - 1] = LAMBDA_MAX + 1;
            values[n - 1] = values[n - 2];

        } else if (_lambda_min > LAMBDA_MIN) {
            n = num_samples / 2 + 1;
            lambdas = new double[n];
            values = new double[n];

            lambdas[0] = LAMBDA_MIN - 1;
            values[0] = samples[1];

            for (int i = 0; i < num_samples / 2; ++i) {
                lambdas[i + 1] = samples[i * 2];
                values[i + 1] = samples[i * 2 + 1];
            }

        } else if (_lambda_max < LAMBDA_MAX) {
            n = num_samples / 2 + 1;
            lambdas = new double[n];
            values = new double[n];

            for (int i = 0; i < num_samples / 2; ++i) {
                lambdas[i] = samples[i * 2];
                values[i] = samples[i * 2 + 1];
            }
            lambdas[n - 1] = LAMBDA_MAX + 1;
            values[n - 1] = values[n - 2];

        } else {
            n = num_samples / 2;

            lambdas = new double[n];
            values = new double[n];

            for (int i = 0; i < num_samples / 2; ++i) {
                lambdas[i] = samples[i * 2];
                values[i] = samples[i * 2 + 1];
            }
        }

        auto spectrum = PiecewiseLinearSpectrum(lambdas, values, n);
        delete lambdas;
        delete values;

        if (normalize) {
            // Normalize to have luminance of 1.
            spectrum.scale(CIE_Y_integral / spectrum.inner_product(*cie_y));
        }

        return spectrum;
    }

    PBRT_GPU
    double operator()(double lambda) const override {
        if (length == 0 || lambda < lambdas[0] || lambda > lambdas[length - 1]) {
            return 0.0;
        }

        int idx = find_interval(length, [&](int i) { return lambdas[i] <= lambda; });
        double t = (lambda - lambdas[idx]) / (lambdas[idx + 1] - lambdas[idx]);

        return lerp(t, values[idx], values[idx + 1]);
    }

    PBRT_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const override {
        std::array<double, NSpectrumSamples> sampled_values{0.0};
        for (int i = 0; i < NSpectrumSamples; ++i) {
            sampled_values[i] = (*this)(lambda[i]);
        }

        return SampledSpectrum(sampled_values);
    }

    PBRT_GPU
    void scale(double s) {
        for (int i = 0; i < length; ++i) {
            values[i] = values[i] * s;
        }
    }

    // private:
    int length = 0;
    double *lambdas = nullptr;
    double *values = nullptr;
};

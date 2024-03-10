#pragma once

#include "pbrt/base/spectrum.h"
#include "pbrt/spectra/constants.h"
#include "pbrt/spectra/black_body_spectrum.h"

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

    PBRT_GPU static DenselySampledSpectrum
    from_piecewise_linear_spectrum(const double *_lambdas, const double *_values, int _length) {
        DenselySampledSpectrum s;
        for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
            s.values[lambda - LAMBDA_MIN] = DenselySampledSpectrum::piecewise_linear_spectrum_eval(
                lambda, _lambdas, _values, _length);
        }

        return s;
    }

    template <typename F>
    PBRT_GPU static DenselySampledSpectrum SampleFunction(F func, int lambda_min = LAMBDA_MIN,
                                                          int lambda_max = LAMBDA_MAX) {
        DenselySampledSpectrum s;
        for (int lambda = lambda_min; lambda <= lambda_max; ++lambda) {
            s.values[lambda - lambda_min] = func(lambda);
        }

        return s;
    }

    PBRT_GPU static DenselySampledSpectrum cie_d(double temperature, const double *cie_s0,
                                                 const double *cie_s1, const double *cie_s2,
                                                 const double *cie_lambda) {
        double cct = temperature * 1.4388f / 1.4380f;
        if (cct < 4000) {
            // CIE D ill-defined, use blackbody
            BlackbodySpectrum bb = BlackbodySpectrum(cct);
            DenselySampledSpectrum blackbody =
                DenselySampledSpectrum::SampleFunction([=](double lambda) { return bb(lambda); });

            return blackbody;
        }

        // Convert CCT to xy
        double x = cct <= 7000 ? -4.607f * 1e9f / fast_powf<3>(cct) + 2.9678f * 1e6f / sqr(cct) +
                                     0.09911f * 1e3f / cct + 0.244063f
                               : -2.0064f * 1e9f / fast_powf<3>(cct) + 1.9018f * 1e6f / sqr(cct) +
                                     0.24748f * 1e3f / cct + 0.23704f;

        double y = -3 * x * x + 2.870f * x - 0.275f;

        // Interpolate D spectrum
        double M = 0.0241f + 0.2562f * x - 0.7341f * y;
        double M1 = (-1.3515f - 1.7703f * x + 5.9114f * y) / M;
        double M2 = (0.0300f - 31.4424f * x + 30.0717f * y) / M;

        double values[nCIES];
        for (int i = 0; i < nCIES; ++i) {
            values[i] = (cie_s0[i] + cie_s1[i] * M1 + cie_s2[i] * M2) * 0.01;
        }

        return DenselySampledSpectrum::from_piecewise_linear_spectrum(cie_lambda, values, nCIES);
    }

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

    PBRT_GPU
    static double piecewise_linear_spectrum_eval(double lambda, const double *lambdas,
                                                 const double *values, int length) {
        if (length == 0 || lambda < lambdas[0] || lambda > lambdas[length - 1]) {
            return 0.0;
        }

        int idx = find_interval(length, [&](int i) { return lambdas[i] <= lambda; });
        double t = (lambda - lambdas[idx]) / (lambdas[idx + 1] - lambdas[idx]);

        return lerp(t, values[idx], values[idx + 1]);
    }
};

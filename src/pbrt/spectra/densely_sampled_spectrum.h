#pragma once

#include "pbrt/spectra/constants.h"
#include "pbrt/spectra/black_body_spectrum.h"

class Spectrum;

namespace {
PBRT_CPU_GPU
double piecewise_linear_spectrum_eval(double lambda, const double *lambdas, const double *values,
                                      uint length) {
    if (lambda < LAMBDA_MIN || lambda > LAMBDA_MAX) {
        return 0.0;
    }

    // TODO: progress 2024/04/01: move this part into init_from_pls_interleaved_samples()?
    if (lambda < lambdas[0] && lambda >= LAMBDA_MIN) {
        return values[0];
    }

    if (lambda > lambdas[length - 1] && lambda <= LAMBDA_MAX) {
        return values[length - 1];
    }
    // TODO: progress 2024/04/01: move this part into init_from_pls_interleaved_samples()?

    uint idx = find_interval(length, [&](uint i) { return lambdas[i] <= lambda; });
    double t = (lambda - lambdas[idx]) / (lambdas[idx + 1] - lambdas[idx]);

    return lerp(t, values[idx], values[idx + 1]);
}
} // namespace

class DenselySampledSpectrum {
  public:
    PBRT_CPU_GPU
    double inner_product(const Spectrum &spectrum) const;

    PBRT_CPU_GPU
    void init_from_pls_lambdas_values(const double *_lambdas, const double *_values, uint _length) {
        for (uint lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
            values[lambda - LAMBDA_MIN] =
                piecewise_linear_spectrum_eval(lambda, _lambdas, _values, _length);
        }
    }

    PBRT_CPU_GPU
    void init_from_pls_interleaved_samples(const double *samples, uint num_samples, bool normalize,
                                           const Spectrum *cie_y);

    template <typename F>
    PBRT_CPU_GPU void init_with_sample_function(F func, uint lambda_min = LAMBDA_MIN,
                                                uint lambda_max = LAMBDA_MAX) {
        for (uint lambda = lambda_min; lambda <= lambda_max; ++lambda) {
            values[lambda - lambda_min] = func(lambda);
        }
    }

    PBRT_CPU_GPU
    void init_cie_d(double temperature, const double *cie_s0, const double *cie_s1,
                    const double *cie_s2, const double *cie_lambda) {
        double cct = temperature * 1.4388f / 1.4380f;
        if (cct < 4000) {
            // CIE D ill-defined, use blackbody
            BlackbodySpectrum bb = BlackbodySpectrum(cct);
            init_with_sample_function([=](double lambda) { return bb(lambda); });
            return;
        }

        // Convert CCT to xy
        double x = cct <= 7000 ? -4.607f * 1e9f / std::pow(cct, 3) + 2.9678f * 1e6f / sqr(cct) +
                                     0.09911f * 1e3f / cct + 0.244063f
                               : -2.0064f * 1e9f / std::pow(cct, 3) + 1.9018f * 1e6f / sqr(cct) +
                                     0.24748f * 1e3f / cct + 0.23704f;

        double y = -3 * x * x + 2.870f * x - 0.275f;

        // Interpolate D spectrum
        double M = 0.0241f + 0.2562f * x - 0.7341f * y;
        double M1 = (-1.3515f - 1.7703f * x + 5.9114f * y) / M;
        double M2 = (0.0300f - 31.4424f * x + 30.0717f * y) / M;

        double _values[nCIES];
        for (int i = 0; i < nCIES; ++i) {
            _values[i] = (cie_s0[i] + cie_s1[i] * M1 + cie_s2[i] * M2) * 0.01;
        }

        init_from_pls_lambdas_values(cie_lambda, _values, nCIES);
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
    double operator()(double lambda) const {
        const int floor = std::floor(lambda);
        const int ceil = std::ceil(lambda);

        if (floor < LAMBDA_MIN || ceil > LAMBDA_MAX) {
            return 0.0;
        }

        return lerp(lambda - floor, values[floor - LAMBDA_MIN], values[ceil - LAMBDA_MIN]);
    }

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const {
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

    PBRT_CPU_GPU
    void scale(double s) {
        for (uint i = 0; i < LAMBDA_RANGE; ++i) {
            values[i] = values[i] * s;
        }
    }

  private:
    double values[LAMBDA_RANGE];
};

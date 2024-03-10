#pragma once

#include "pbrt/base/spectrum.h"

// Spectrum Function Declarations
PBRT_CPU_GPU inline double Blackbody(double lambda, double T) {
    if (T <= 0) {
        return 0.0;
    }

    const double c = 299792458.f;
    const double h = 6.62606957e-34f;
    const double kb = 1.3806488e-23f;
    // Return emitted radiance for blackbody at wavelength _lambda_
    double l = lambda * 1e-9f;
    double Le = (2 * h * c * c) / (fast_powf<5>(l) * (std::exp((h * c) / (l * kb * T)) - 1.0));
    return Le;
}

class BlackbodySpectrum {
  public:
    // BlackbodySpectrum Public Methods
    PBRT_CPU_GPU
    BlackbodySpectrum(double T) : T(T) {
        // Compute blackbody normalization constant for given temperature
        double lambdaMax = 2.8977721e-3f / T;
        normalizationFactor = 1 / Blackbody(lambdaMax * 1e9f, T);
    }

    PBRT_CPU_GPU
    double operator()(double lambda) const {
        return Blackbody(lambda, T) * normalizationFactor;
    }

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const {
        std::array<double, NSpectrumSamples> values;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            values[i] = Blackbody(lambda[i], T) * normalizationFactor;
        }

        return SampledSpectrum(values);
    }

  private:
    double T;
    double normalizationFactor;
};

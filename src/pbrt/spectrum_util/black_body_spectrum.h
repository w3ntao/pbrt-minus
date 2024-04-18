#pragma once

#include "pbrt/util/macro.h"
#include "sampled_spectrum.h"
#include "sampled_wavelengths.h"

// Spectrum Function Declarations
PBRT_CPU_GPU inline FloatType Blackbody(FloatType lambda, FloatType T) {
    if (T <= 0) {
        return 0.0;
    }

    const FloatType c = 299792458.f;
    const FloatType h = 6.62606957e-34f;
    const FloatType kb = 1.3806488e-23f;
    // Return emitted radiance for blackbody at wavelength _lambda_
    FloatType l = lambda * 1e-9f;

    FloatType Le = (2 * h * c * c) / (std::pow(l, 5) * (std::exp((h * c) / (l * kb * T)) - 1.0));
    return Le;
}

class BlackbodySpectrum {
  public:
    // BlackbodySpectrum Public Methods
    PBRT_CPU_GPU
    BlackbodySpectrum(FloatType T) : T(T) {
        // Compute blackbody normalization constant for given temperature
        FloatType lambdaMax = 2.8977721e-3f / T;
        normalizationFactor = 1 / Blackbody(lambdaMax * 1e9f, T);
    }

    PBRT_CPU_GPU
    FloatType operator()(FloatType lambda) const {
        return Blackbody(lambda, T) * normalizationFactor;
    }

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const {
        FloatType values[NSpectrumSamples];

        for (int i = 0; i < NSpectrumSamples; ++i) {
            values[i] = Blackbody(lambda[i], T) * normalizationFactor;
        }

        return SampledSpectrum(values);
    }

  private:
    FloatType T;
    FloatType normalizationFactor;
};

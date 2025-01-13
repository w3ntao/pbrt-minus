#pragma once

#include <pbrt/spectrum_util/sampled_spectrum.h>
#include <pbrt/spectrum_util/sampled_wavelengths.h>
#include <pbrt/gpu/macro.h>

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
    BlackbodySpectrum(FloatType _T) {
        init(_T);
    }

    PBRT_CPU_GPU
    void init(FloatType _T) {
        // Compute blackbody normalization constant for given temperature
        T = _T;
        FloatType lambdaMax = 2.8977721e-3f / _T;
        normalization_factor = 1 / Blackbody(lambdaMax * 1e9f, _T);
    }

    PBRT_CPU_GPU FloatType operator()(FloatType lambda) const {
        return Blackbody(lambda, T) * normalization_factor;
    }

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const {
        SampledSpectrum result;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            result[i] = Blackbody(lambda[i], T) * normalization_factor;
        }

        return result;
    }

  private:
    FloatType T;
    FloatType normalization_factor;
};

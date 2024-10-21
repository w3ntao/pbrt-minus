#pragma once

#include "pbrt/spectrum_util/spectrum_constants_cie.h"
#include "pbrt/spectrum_util/xyz.h"

// TODO: move those function into sampled_spectrum.cpp

class Spectrum;
class SampledWavelengths;

class SampledSpectrum {
  public:
    PBRT_CPU_GPU SampledSpectrum() {
        for (uint idx = 0; idx < NSpectrumSamples; ++idx) {
            values[idx] = NAN;
        }
    }

    PBRT_CPU_GPU
    SampledSpectrum(const FloatType val) {
        for (uint idx = 0; idx < NSpectrumSamples; ++idx) {
            values[idx] = val;
        }
    }

    PBRT_CPU_GPU
    bool has_nan() const {
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            if (isnan(values[i])) {
                return true;
            }
        }

        return false;
    }

    PBRT_CPU_GPU
    SampledSpectrum sqrt() const {
        SampledSpectrum result;
        for (uint idx = 0; idx < NSpectrumSamples; ++idx) {
            result[idx] = std::sqrt(values[idx]);
        }

        return result;
    }

    PBRT_CPU_GPU
    SampledSpectrum clamp(FloatType low, FloatType high) const {
        SampledSpectrum result;
        for (uint idx = 0; idx < NSpectrumSamples; ++idx) {
            result[idx] = ::clamp(values[idx], low, high);
        }

        return result;
    }

    PBRT_CPU_GPU
    FloatType y(const SampledWavelengths &lambda, const Spectrum *cie_y) const;

    PBRT_CPU_GPU
    inline FloatType operator[](uint8_t i) const {
        return values[i];
    }

    PBRT_CPU_GPU
    inline FloatType &operator[](uint8_t i) {
        return values[i];
    }

    PBRT_CPU_GPU
    SampledSpectrum operator+(const SampledSpectrum &s) const {
        SampledSpectrum sum;
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            sum[i] = values[i] + s.values[i];
        }

        return sum;
    }

    PBRT_CPU_GPU
    void operator+=(const SampledSpectrum &s) {
        *this = *this + s;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator-(const SampledSpectrum &s) const {
        SampledSpectrum difference;
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            difference[i] = values[i] - s.values[i];
        }

        return difference;
    }

    PBRT_CPU_GPU
    void operator-=(const SampledSpectrum &s) {
        *this = *this - s;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator*(const SampledSpectrum &s) const {
        SampledSpectrum product;
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            product[i] = values[i] * s.values[i];
        }

        return product;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator*(FloatType a) const {
        SampledSpectrum product;
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            product[i] = values[i] * a;
        }

        return product;
    }

    PBRT_CPU_GPU
    friend SampledSpectrum operator*(FloatType a, const SampledSpectrum &s) {
        return s * a;
    }

    PBRT_CPU_GPU
    void operator*=(const SampledSpectrum &s) {
        *this = *this * s;
    }

    PBRT_CPU_GPU
    void operator*=(FloatType a) {
        *this = *this * a;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator/(const SampledSpectrum &s) const {
        SampledSpectrum quotient;
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            quotient[i] = values[i] / s.values[i];
        }

        return quotient;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator/(FloatType a) const {
        SampledSpectrum quotient;
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            quotient[i] = values[i] / a;
        }

        return quotient;
    }

    PBRT_CPU_GPU
    void operator/=(const SampledSpectrum &s) {
        *this = *this / s;
    }

    PBRT_CPU_GPU
    void operator/=(FloatType a) {
        *this = *this / a;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator-() const {
        SampledSpectrum result;

        for (uint i = 0; i < NSpectrumSamples; ++i) {
            result[i] = -values[i];
        }

        return result;
    }

    PBRT_CPU_GPU
    bool operator==(const SampledSpectrum &s) const {
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            if (values[i] != s.values[i]) {
                return false;
            }
        }

        return true;
    }

    PBRT_CPU_GPU
    bool operator!=(const SampledSpectrum &s) const {
        return !(*this == s);
    }

    PBRT_CPU_GPU
    bool is_positive() const {
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            if (values[i] > 0) {
                return true;
            }
        }

        return false;
    }

    PBRT_CPU_GPU
    FloatType min_component_value() const {
        FloatType m = values[0];
        for (uint i = 1; i < NSpectrumSamples; ++i) {
            m = std::min(m, values[i]);
        }

        return m;
    }

    PBRT_CPU_GPU
    FloatType max_component_value() const {
        FloatType m = values[0];
        for (uint i = 1; i < NSpectrumSamples; ++i) {
            m = std::max(m, values[i]);
        }
        return m;
    }

    PBRT_CPU_GPU
    FloatType average() const {
        FloatType sum = 0;
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            sum += values[i];
        }

        return sum / NSpectrumSamples;
    }

    PBRT_CPU_GPU void print() const {
        printf("[ ");
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            printf("%f, ", values[i]);
        }
        printf("]\n");
    }

    PBRT_CPU_GPU SampledSpectrum safe_div(const SampledSpectrum &divisor) const {
        SampledSpectrum quotient;
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            quotient[i] = divisor[i] == 0.0 ? 0.0 : values[i] / divisor[i];
        }

        return quotient;
    }

  private:
    FloatType values[NSpectrumSamples];
};

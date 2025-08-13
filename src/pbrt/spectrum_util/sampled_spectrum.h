#pragma once

#include <pbrt/gpu/macro.h>
#include <pbrt/spectrum_util/spectrum_constants_cie.h>
#include <pbrt/util/math.h>

class Spectrum;
class SampledWavelengths;

class SampledSpectrum {
  public:
    PBRT_CPU_GPU
    SampledSpectrum() {
        for (int idx = 0; idx < NSpectrumSamples; ++idx) {
            values[idx] = 0;
        }
    }

    PBRT_CPU_GPU
    SampledSpectrum(const Real val) {
        for (int idx = 0; idx < NSpectrumSamples; ++idx) {
            values[idx] = val;
        }
    }

    PBRT_CPU_GPU
    bool has_nan() const {
        for (int i = 0; i < NSpectrumSamples; ++i) {
            if (isnan(values[i]) || isinf(values[i])) {
                return true;
            }
        }

        return false;
    }

    PBRT_CPU_GPU
    static SampledSpectrum exp(const SampledSpectrum &s) {
        SampledSpectrum result;
        for (int idx = 0; idx < NSpectrumSamples; ++idx) {
            result[idx] = std::exp(s[idx]);
        }

        return result;
    }

    PBRT_CPU_GPU
    SampledSpectrum sqrt() const {
        SampledSpectrum result;
        for (int idx = 0; idx < NSpectrumSamples; ++idx) {
            result[idx] = std::sqrt(values[idx]);
        }

        return result;
    }

    PBRT_CPU_GPU
    SampledSpectrum clamp(Real low, Real high) const {
        SampledSpectrum result;
        for (int idx = 0; idx < NSpectrumSamples; ++idx) {
            result[idx] = ::clamp(values[idx], low, high);
        }

        return result;
    }

    PBRT_CPU_GPU
    Real y(const SampledWavelengths &lambda, const Spectrum *cie_y) const;

    PBRT_CPU_GPU
    Real operator[](uint8_t i) const {
        return values[i];
    }

    PBRT_CPU_GPU
    Real &operator[](uint8_t i) {
        return values[i];
    }

    PBRT_CPU_GPU
    SampledSpectrum operator+(const SampledSpectrum &s) const {
        SampledSpectrum sum;
        for (int i = 0; i < NSpectrumSamples; ++i) {
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
        for (int i = 0; i < NSpectrumSamples; ++i) {
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
        for (int i = 0; i < NSpectrumSamples; ++i) {
            product[i] = values[i] * s.values[i];
        }

        return product;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator*(Real a) const {
        SampledSpectrum product;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            product[i] = values[i] * a;
        }

        return product;
    }

    PBRT_CPU_GPU
    friend SampledSpectrum operator*(Real a, const SampledSpectrum &s) {
        return s * a;
    }

    PBRT_CPU_GPU
    void operator*=(const SampledSpectrum &s) {
        *this = *this * s;
    }

    PBRT_CPU_GPU
    void operator*=(Real a) {
        *this = *this * a;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator/(const SampledSpectrum &s) const {
        SampledSpectrum quotient;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            quotient[i] = values[i] / s.values[i];
        }

        return quotient;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator/(Real a) const {
        SampledSpectrum quotient;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            quotient[i] = values[i] / a;
        }

        return quotient;
    }

    PBRT_CPU_GPU
    void operator/=(const SampledSpectrum &s) {
        *this = *this / s;
    }

    PBRT_CPU_GPU
    void operator/=(Real a) {
        *this = *this / a;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator-() const {
        SampledSpectrum result;

        for (int i = 0; i < NSpectrumSamples; ++i) {
            result[i] = -values[i];
        }

        return result;
    }

    PBRT_CPU_GPU
    bool operator==(const SampledSpectrum &s) const {
        for (int i = 0; i < NSpectrumSamples; ++i) {
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
        for (int i = 0; i < NSpectrumSamples; ++i) {
            if (values[i] > 0) {
                return true;
            }
        }

        return false;
    }

    PBRT_CPU_GPU
    Real min_component_value() const {
        Real m = values[0];
        for (int i = 1; i < NSpectrumSamples; ++i) {
            m = std::min(m, values[i]);
        }

        return m;
    }

    PBRT_CPU_GPU
    Real max_component_value() const {
        Real m = values[0];
        for (int i = 1; i < NSpectrumSamples; ++i) {
            m = std::max(m, values[i]);
        }
        return m;
    }

    PBRT_CPU_GPU
    Real average() const {
        Real sum = 0;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            sum += values[i];
        }

        return sum / NSpectrumSamples;
    }

    PBRT_CPU_GPU void print() const {
        printf("[ ");
        for (int i = 0; i < NSpectrumSamples; ++i) {
            printf("%f, ", values[i]);
        }
        printf("]\n");
    }

    PBRT_CPU_GPU SampledSpectrum safe_div(const SampledSpectrum &divisor) const {
        SampledSpectrum quotient;
        for (int i = 0; i < NSpectrumSamples; ++i) {
            quotient[i] = divisor[i] == 0.0 ? 0.0 : values[i] / divisor[i];
        }

        return quotient;
    }

  private:
    Real values[NSpectrumSamples];
};

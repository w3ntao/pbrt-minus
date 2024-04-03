#pragma once

#include "pbrt/spectra/xyz.h"
#include "pbrt/spectra/constants.h"

class SampledSpectrum {
  public:
    PBRT_CPU_GPU explicit SampledSpectrum(const double _values[NSpectrumSamples]) {
        for (uint idx = 0; idx < NSpectrumSamples; ++idx) {
            values[idx] = _values[idx];
        }
    }

    PBRT_CPU_GPU
    static SampledSpectrum same_value(double c) {
        double _values[NSpectrumSamples];
        for (uint i = 0; i < NSpectrumSamples; ++i) {
            _values[i] = c;
        }
        return SampledSpectrum(_values);
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
    double operator[](uint8_t i) const {
        return values[i];
    }

    PBRT_CPU_GPU
    double &operator[](uint8_t i) {
        return values[i];
    }

    PBRT_CPU_GPU
    SampledSpectrum operator+(const SampledSpectrum &s) const {
        double sum[NSpectrumSamples];

        for (uint i = 0; i < NSpectrumSamples; ++i) {
            sum[i] = values[i] + s.values[i];
        }

        return SampledSpectrum(sum);
    }

    PBRT_CPU_GPU
    void operator+=(const SampledSpectrum &s) {
        *this = *this + s;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator-(const SampledSpectrum &s) const {
        double difference[NSpectrumSamples];

        for (uint i = 0; i < NSpectrumSamples; ++i) {
            difference[i] = values[i] - s.values[i];
        }

        return SampledSpectrum(difference);
    }

    PBRT_CPU_GPU
    void operator-=(const SampledSpectrum &s) {
        *this = *this - s;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator*(const SampledSpectrum &s) const {
        double product[NSpectrumSamples];

        for (uint i = 0; i < NSpectrumSamples; ++i) {
            product[i] = values[i] * s.values[i];
        }

        return SampledSpectrum(product);
    }

    PBRT_CPU_GPU
    SampledSpectrum operator*(double a) const {
        double product[NSpectrumSamples];

        for (uint i = 0; i < NSpectrumSamples; ++i) {
            product[i] = values[i] * a;
        }

        return SampledSpectrum(product);
    }

    PBRT_CPU_GPU
    friend SampledSpectrum operator*(double a, const SampledSpectrum &s) {
        return s * a;
    }

    PBRT_CPU_GPU
    void operator*=(const SampledSpectrum &s) {
        *this = *this * s;
    }

    PBRT_CPU_GPU
    void operator*=(double a) {
        *this = *this * a;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator/(const SampledSpectrum &s) const {
        double quotient[NSpectrumSamples];

        for (uint i = 0; i < NSpectrumSamples; ++i) {
            quotient[i] = values[i] / s.values[i];
        }

        return SampledSpectrum(quotient);
    }

    PBRT_CPU_GPU
    SampledSpectrum operator/(double a) const {
        double quotient[NSpectrumSamples];

        for (uint i = 0; i < NSpectrumSamples; ++i) {
            quotient[i] = values[i] / a;
        }

        return SampledSpectrum(quotient);
    }

    PBRT_CPU_GPU
    void operator/=(const SampledSpectrum &s) {
        *this = *this / s;
    }

    PBRT_CPU_GPU
    void operator/=(double a) {
        *this = *this / a;
    }

    PBRT_CPU_GPU
    SampledSpectrum operator-() const {
        double ret[NSpectrumSamples];

        for (uint i = 0; i < NSpectrumSamples; ++i) {
            ret[i] = -values[i];
        }

        return SampledSpectrum(ret);
    }

    PBRT_CPU_GPU
    bool operator==(const SampledSpectrum &s) const {
        return values == s.values;
    }

    PBRT_CPU_GPU
    bool operator!=(const SampledSpectrum &s) const {
        return values != s.values;
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
    double min_component_value() const {
        double m = values[0];
        for (uint i = 1; i < NSpectrumSamples; ++i) {
            m = std::min(m, values[i]);
        }

        return m;
    }

    PBRT_CPU_GPU
    double max_component_value() const {
        double m = values[0];
        for (uint i = 1; i < NSpectrumSamples; ++i) {
            m = std::max(m, values[i]);
        }
        return m;
    }

    PBRT_CPU_GPU
    double average() const {
        double sum = 0;
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
        double quotient[NSpectrumSamples];

        for (uint i = 0; i < NSpectrumSamples; ++i) {
            quotient[i] = divisor[i] == 0.0 ? 0.0 : values[i] / divisor[i];
        }

        return SampledSpectrum(quotient);
    }

  private:
    double values[NSpectrumSamples];
};

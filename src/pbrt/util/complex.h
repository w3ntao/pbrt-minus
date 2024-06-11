#pragma once

#include "pbrt/util/macro.h"

namespace pstd {

template <typename T>
struct complex {
    PBRT_CPU_GPU complex(T _re) : re(_re), im(0) {}
    PBRT_CPU_GPU complex(T _re, T _im) : re(_re), im(_im) {}

    PBRT_CPU_GPU T norm() const {
        return re * re + im * im;
    }

    PBRT_CPU_GPU T abs() const {
        return std::sqrt(norm());
    }

    PBRT_CPU_GPU complex<T> sqrt() const {
        T n = this->abs();
        T t1 = std::sqrt(T(.5) * (n + std::abs(re)));
        T t2 = T(.5) * im / t1;

        if (n == 0) {
            return 0;
        }

        if (re >= 0) {
            return {t1, t2};
        } else {
            return {std::abs(t2), pstd::copysign(t1, im)};
        }
    }

    PBRT_CPU_GPU complex operator-() const {
        return {-re, -im};
    }

    PBRT_CPU_GPU complex operator+(complex z) const {
        return {re + z.re, im + z.im};
    }

    PBRT_CPU_GPU complex operator-(complex z) const {
        return {re - z.re, im - z.im};
    }

    PBRT_CPU_GPU complex operator*(complex z) const {
        return {re * z.re - im * z.im, re * z.im + im * z.re};
    }

    PBRT_CPU_GPU complex operator/(complex z) const {
        T scale = 1 / (z.re * z.re + z.im * z.im);
        return {scale * (re * z.re + im * z.im), scale * (im * z.re - re * z.im)};
    }

    friend PBRT_CPU_GPU complex operator+(T value, complex z) {
        return complex(value) + z;
    }

    friend PBRT_CPU_GPU complex operator-(T value, complex z) {
        return complex(value) - z;
    }

    friend PBRT_CPU_GPU complex operator*(T value, complex z) {
        return complex(value) * z;
    }

    friend PBRT_CPU_GPU complex operator/(T value, complex z) {
        return complex(value) / z;
    }

    T re, im;
};

} // namespace pstd

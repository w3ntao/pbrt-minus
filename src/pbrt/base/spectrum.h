#pragma once

#include "pbrt/spectra/sampled_wavelengths.h"
#include "pbrt/spectra/xyz.h"

class Spectrum {
  public:
    PBRT_GPU
    virtual ~Spectrum() {}

    PBRT_GPU
    virtual double operator()(double lambda) const = 0;

    PBRT_GPU
    double inner_product(const Spectrum &spectrum) const {
        double sum = 0;
        for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
            sum += (*this)(lambda)*spectrum(lambda);
        }

        return sum;
    }

    PBRT_GPU
    virtual SampledSpectrum sample(const SampledWavelengths &lambda) const = 0;

    PBRT_GPU
    XYZ to_xyz(const std::array<const Spectrum *, 3> &cie_xyz) const {
        auto x = cie_xyz[0];
        auto y = cie_xyz[1];
        auto z = cie_xyz[2];

        return XYZ(inner_product(*x), inner_product(*y), inner_product(*z)) / CIE_Y_integral;
    }

    PBRT_GPU double to_photometric(const Spectrum &cie_y) const {
        return inner_product(cie_y);
    }
};

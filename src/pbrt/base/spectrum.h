#pragma once

#include <vector>

#include "pbrt/spectrum_util/sampled_wavelengths.h"
#include "pbrt/spectrum_util/xyz.h"

class RGB;
class RGBColorSpace;

class BlackbodySpectrum;
class ConstantSpectrum;
class DenselySampledSpectrum;
class RGBAlbedoSpectrum;
class RGBIlluminantSpectrum;
class RGBUnboundedSpectrum;
class PiecewiseLinearSpectrum;

enum class SpectrumType {
    Illuminant,
    Albedo,
    Unbounded,
};

class Spectrum {
  public:
    enum class Type {
        black_body,
        constant,
        densely_sampled,
        rgb_albedo,
        rgb_illuminant,
        rgb_unbounded,
        piecewise_linear,
    };

    static const Spectrum *create_black_body(FloatType T, std::vector<void *> &gpu_dynamic_pointer);

    static const Spectrum *create_cie_d(FloatType temperature, const FloatType *cie_s0,
                                        const FloatType *cie_s1, const FloatType *cie_s2,
                                        const FloatType *cie_lambda,
                                        std::vector<void *> &gpu_dynamic_pointer);

    static const Spectrum *create_constant_spectrum(FloatType val,
                                                    std::vector<void *> &gpu_dynamic_pointers);

    static const Spectrum *create_from_rgb(const RGB &val, SpectrumType spectrum_type,
                                           const RGBColorSpace *color_space,
                                           std::vector<void *> &gpu_dynamic_pointers);

    static const Spectrum *create_piecewise_linear_spectrum_from_lambdas_and_values(
        const std::vector<FloatType> &cpu_lambdas, const std::vector<FloatType> &cpu_values,
        std::vector<void *> &gpu_dynamic_pointers);

    static const Spectrum *
    create_piecewise_linear_spectrum_from_interleaved(const std::vector<FloatType> &samples,
                                                      bool normalize, const Spectrum *cie_y,
                                                      std::vector<void *> &gpu_dynamic_pointers);
    PBRT_CPU_GPU
    void init(const BlackbodySpectrum *black_body_spectrum);

    PBRT_CPU_GPU
    void init(const ConstantSpectrum *constant_spectrum);

    PBRT_CPU_GPU
    void init(const DenselySampledSpectrum *densely_sampled_spectrum);

    PBRT_CPU_GPU
    void init(const RGBAlbedoSpectrum *rgb_albedo_spectrum);

    PBRT_CPU_GPU
    void init(const RGBIlluminantSpectrum *rgb_illuminant_spectrum);

    PBRT_CPU_GPU
    void init(const RGBUnboundedSpectrum *rgb_unbounded_spectrum);

    PBRT_CPU_GPU
    void init(const PiecewiseLinearSpectrum *piecewise_linear_spectrum);

    PBRT_CPU_GPU bool is_constant_spectrum() const {
        return type == Type::constant;
    }

    PBRT_CPU_GPU FloatType operator()(FloatType lambda) const;

    PBRT_CPU_GPU
    FloatType inner_product(const Spectrum *const spectrum) const {
        FloatType sum = 0;
        for (int lambda = LAMBDA_MIN; lambda <= LAMBDA_MAX; ++lambda) {
            sum += (*this)(lambda) * (*spectrum)(lambda);
        }

        return sum;
    }

    PBRT_CPU_GPU
    SampledSpectrum sample(const SampledWavelengths &lambda) const;

    PBRT_CPU_GPU
    XYZ to_xyz(const Spectrum *const cie_xyz[3]) const {
        return XYZ(inner_product(cie_xyz[0]), inner_product(cie_xyz[1]),
                   inner_product(cie_xyz[2])) /
               CIE_Y_integral;
    }

    PBRT_CPU_GPU
    FloatType to_photometric(const Spectrum *cie_y) const;

  private:
    Type type;
    const void *ptr;
};

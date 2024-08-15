#include "pbrt/base/spectrum.h"

#include "pbrt/spectra/black_body_spectrum.h"
#include "pbrt/spectra/constant_spectrum.h"
#include "pbrt/spectra/densely_sampled_spectrum.h"
#include "pbrt/spectra/rgb_albedo_spectrum.h"
#include "pbrt/spectra/rgb_illuminant_spectrum.h"
#include "pbrt/spectra/rgb_unbounded_spectrum.h"
#include "pbrt/spectra/piecewise_linear_spectrum.h"

const Spectrum *Spectrum::create_black_body(FloatType T, std::vector<void *> &gpu_dynamic_pointer) {
    BlackbodySpectrum *black_body;
    CHECK_CUDA_ERROR(cudaMallocManaged(&black_body, sizeof(BlackbodySpectrum)));
    Spectrum *spectrum;
    CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum, sizeof(Spectrum)));
    gpu_dynamic_pointer.push_back(black_body);
    gpu_dynamic_pointer.push_back(spectrum);

    black_body->init(T);
    spectrum->init(black_body);

    return spectrum;
}

const Spectrum *Spectrum::create_cie_d(FloatType temperature, const FloatType *cie_s0,
                                       const FloatType *cie_s1, const FloatType *cie_s2,
                                       const FloatType *cie_lambda,
                                       std::vector<void *> &gpu_dynamic_pointer) {
    FloatType cct = temperature * 1.4388f / 1.4380f;
    if (cct < 4000) {
        // CIE D ill-defined, use blackbody
        BlackbodySpectrum bb = BlackbodySpectrum(cct);

        DenselySampledSpectrum *densely_sampled_spectrum;
        CHECK_CUDA_ERROR(
            cudaMallocManaged(&densely_sampled_spectrum, sizeof(DenselySampledSpectrum)));
        densely_sampled_spectrum->init_with_sample_function(
            [=](FloatType lambda) { return bb(lambda); });

        Spectrum *spectrum;
        CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum, sizeof(Spectrum)));
        spectrum->init(densely_sampled_spectrum);

        gpu_dynamic_pointer.push_back(densely_sampled_spectrum);
        gpu_dynamic_pointer.push_back(spectrum);

        return spectrum;
    }

    // Convert CCT to xy
    FloatType x = cct <= 7000 ? -4.607f * 1e9f / std::pow(cct, 3) + 2.9678f * 1e6f / sqr(cct) +
                                    0.09911f * 1e3f / cct + 0.244063f
                              : -2.0064f * 1e9f / std::pow(cct, 3) + 1.9018f * 1e6f / sqr(cct) +
                                    0.24748f * 1e3f / cct + 0.23704f;

    FloatType y = -3 * x * x + 2.870f * x - 0.275f;

    // Interpolate D spectrum
    FloatType M = 0.0241f + 0.2562f * x - 0.7341f * y;
    FloatType M1 = (-1.3515f - 1.7703f * x + 5.9114f * y) / M;
    FloatType M2 = (0.0300f - 31.4424f * x + 30.0717f * y) / M;

    std::vector<FloatType> cpu_cie_lambdas(nCIES);
    std::vector<FloatType> cpu_values(nCIES);

    for (uint idx = 0; idx < nCIES; ++idx) {
        cpu_cie_lambdas.push_back(cie_lambda[idx]);
        cpu_values.push_back((cie_s0[idx] + cie_s1[idx] * M1 + cie_s2[idx] * M2) * 0.01);
    }

    return Spectrum::create_piecewise_linear_spectrum_from_lambdas_and_values(
        cpu_cie_lambdas, cpu_values, gpu_dynamic_pointer);
}

const Spectrum *Spectrum::create_constant_spectrum(FloatType val,
                                                   std::vector<void *> &gpu_dynamic_pointers) {
    ConstantSpectrum *constant_spectrum;
    Spectrum *spectrum;
    CHECK_CUDA_ERROR(cudaMallocManaged(&constant_spectrum, sizeof(ConstantSpectrum)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum, sizeof(Spectrum)));
    gpu_dynamic_pointers.push_back(constant_spectrum);
    gpu_dynamic_pointers.push_back(spectrum);

    constant_spectrum->init(val);
    spectrum->init(constant_spectrum);

    return spectrum;
}

const Spectrum *Spectrum::create_from_rgb(const RGB &val, SpectrumType spectrum_type,
                                          const RGBColorSpace *color_space,
                                          std::vector<void *> &gpu_dynamic_pointers) {
    switch (spectrum_type) {
    case (SpectrumType::Albedo): {
        if (val.r > 1 || val.g > 1 || val.b > 1) {
            REPORT_FATAL_ERROR();
        }

        RGBAlbedoSpectrum *rgb_albedo_spectrum;
        Spectrum *spectrum;
        CHECK_CUDA_ERROR(cudaMallocManaged(&rgb_albedo_spectrum, sizeof(RGBAlbedoSpectrum)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum, sizeof(Spectrum)));
        gpu_dynamic_pointers.push_back(rgb_albedo_spectrum);
        gpu_dynamic_pointers.push_back(spectrum);

        rgb_albedo_spectrum->init(val, color_space);
        spectrum->init(rgb_albedo_spectrum);

        return spectrum;
    }

    case (SpectrumType::Illuminant): {
        RGBIlluminantSpectrum *rgb_illuminant_spectrum;
        Spectrum *spectrum;
        CHECK_CUDA_ERROR(
            cudaMallocManaged(&rgb_illuminant_spectrum, sizeof(RGBIlluminantSpectrum)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum, sizeof(Spectrum)));

        gpu_dynamic_pointers.push_back(rgb_illuminant_spectrum);
        gpu_dynamic_pointers.push_back(spectrum);

        rgb_illuminant_spectrum->init(val, color_space);
        spectrum->init(rgb_illuminant_spectrum);

        return spectrum;
    }

    case (SpectrumType::Unbounded): {
        RGBUnboundedSpectrum *rgb_unbounded_spectrum;
        Spectrum *spectrum;
        CHECK_CUDA_ERROR(cudaMallocManaged(&rgb_unbounded_spectrum, sizeof(RGBUnboundedSpectrum)));
        CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum, sizeof(Spectrum)));

        gpu_dynamic_pointers.push_back(rgb_unbounded_spectrum);
        gpu_dynamic_pointers.push_back(spectrum);

        rgb_unbounded_spectrum->init(val, color_space);
        spectrum->init(rgb_unbounded_spectrum);

        return spectrum;
    }
    }

    printf("\n SpectrumType `%d` not implemented\n\n", spectrum_type);

    REPORT_FATAL_ERROR();
    return nullptr;
}

const Spectrum *Spectrum::create_piecewise_linear_spectrum_from_lambdas_and_values(
    const std::vector<FloatType> &cpu_lambdas, const std::vector<FloatType> &cpu_values,
    std::vector<void *> &gpu_dynamic_pointers) {
    PiecewiseLinearSpectrum *piecewise_linear_spectrum;
    CHECK_CUDA_ERROR(
        cudaMallocManaged(&piecewise_linear_spectrum, sizeof(PiecewiseLinearSpectrum)));
    piecewise_linear_spectrum->init_from_lambdas_values(cpu_lambdas, cpu_values,
                                                        gpu_dynamic_pointers);

    Spectrum *spectrum;
    CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum, sizeof(Spectrum)));
    spectrum->init(piecewise_linear_spectrum);

    gpu_dynamic_pointers.push_back(piecewise_linear_spectrum);
    gpu_dynamic_pointers.push_back(spectrum);

    return spectrum;
}

const Spectrum *Spectrum::create_piecewise_linear_spectrum_from_interleaved(
    const std::vector<FloatType> &samples, bool normalize, const Spectrum *cie_y,
    std::vector<void *> &gpu_dynamic_pointers) {

    PiecewiseLinearSpectrum *piecewise_linear_spectrum;
    CHECK_CUDA_ERROR(
        cudaMallocManaged(&piecewise_linear_spectrum, sizeof(PiecewiseLinearSpectrum)));

    piecewise_linear_spectrum->init_from_interleaved(samples, normalize, cie_y,
                                                     gpu_dynamic_pointers);

    Spectrum *spectrum;
    CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum, sizeof(Spectrum)));
    spectrum->init(piecewise_linear_spectrum);

    gpu_dynamic_pointers.push_back(piecewise_linear_spectrum);
    gpu_dynamic_pointers.push_back(spectrum);

    return spectrum;
}

PBRT_CPU_GPU
void Spectrum::init(const BlackbodySpectrum *black_body_spectrum) {
    type = Type::black_body;
    ptr = black_body_spectrum;
}

PBRT_CPU_GPU
void Spectrum::init(const ConstantSpectrum *constant_spectrum) {
    type = Type::constant;
    ptr = constant_spectrum;
}

PBRT_CPU_GPU
void Spectrum::init(const DenselySampledSpectrum *densely_sampled_spectrum) {
    type = Type::densely_sampled;
    ptr = densely_sampled_spectrum;
}

PBRT_CPU_GPU
void Spectrum::init(const RGBAlbedoSpectrum *rgb_albedo_spectrum) {
    type = Type::rgb_albedo;
    ptr = rgb_albedo_spectrum;
}

PBRT_CPU_GPU
void Spectrum::init(const RGBIlluminantSpectrum *rgb_illuminant_spectrum) {
    type = Type::rgb_illuminant;
    ptr = rgb_illuminant_spectrum;
}

PBRT_CPU_GPU
void Spectrum::init(const RGBUnboundedSpectrum *rgb_unbounded_spectrum) {
    type = Type::rgb_unbounded;
    ptr = rgb_unbounded_spectrum;
}

PBRT_CPU_GPU
void Spectrum::init(const PiecewiseLinearSpectrum *piecewise_linear_spectrum) {
    type = Type::piecewise_linear;
    ptr = piecewise_linear_spectrum;
}

PBRT_CPU_GPU
FloatType Spectrum::operator()(FloatType lambda) const {
    switch (type) {
    case (Type::black_body): {
        return ((BlackbodySpectrum *)ptr)->operator()(lambda);
    }

    case (Type::constant): {
        return ((ConstantSpectrum *)ptr)->operator()(lambda);
    }

    case (Type::densely_sampled): {
        return ((DenselySampledSpectrum *)ptr)->operator()(lambda);
    }

    case (Type::piecewise_linear): {
        return ((PiecewiseLinearSpectrum *)ptr)->operator()(lambda);
    }

    case (Type::rgb_albedo): {
        return ((RGBAlbedoSpectrum *)ptr)->operator()(lambda);
    }

    case (Type::rgb_illuminant): {
        return ((RGBIlluminantSpectrum *)ptr)->operator()(lambda);
    }

    case (Type::rgb_unbounded): {
        return ((RGBUnboundedSpectrum *)ptr)->operator()(lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

PBRT_CPU_GPU
FloatType Spectrum::to_photometric(const Spectrum *cie_y) const {
    if (type == Type::rgb_illuminant) {
        return ((RGBIlluminantSpectrum *)ptr)->to_photometric(cie_y);
    }

    return inner_product(cie_y);
}

PBRT_CPU_GPU
SampledSpectrum Spectrum::sample(const SampledWavelengths &lambda) const {
    switch (type) {
    case (Type::black_body): {
        return ((BlackbodySpectrum *)ptr)->sample(lambda);
    }

    case (Type::constant): {
        return ((ConstantSpectrum *)ptr)->sample(lambda);
    }

    case (Type::densely_sampled): {
        return ((DenselySampledSpectrum *)ptr)->sample(lambda);
    }

    case (Type::piecewise_linear): {
        return ((PiecewiseLinearSpectrum *)ptr)->sample(lambda);
    }

    case (Type::rgb_albedo): {
        return ((RGBAlbedoSpectrum *)ptr)->sample(lambda);
    }

    case (Type::rgb_illuminant): {
        return ((RGBIlluminantSpectrum *)ptr)->sample(lambda);
    }

    case (Type::rgb_unbounded): {
        return ((RGBUnboundedSpectrum *)ptr)->sample(lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return SampledSpectrum(NAN);
}

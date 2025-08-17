#include <pbrt/base/spectrum.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/spectra/black_body_spectrum.h>
#include <pbrt/spectra/constant_spectrum.h>
#include <pbrt/spectra/densely_sampled_spectrum.h>
#include <pbrt/spectra/piecewise_linear_spectrum.h>
#include <pbrt/spectra/rgb_albedo_spectrum.h>
#include <pbrt/spectra/rgb_illuminant_spectrum.h>
#include <pbrt/spectra/rgb_unbounded_spectrum.h>

const Spectrum *Spectrum::create_black_body(Real temperature, GPUMemoryAllocator &allocator) {
    auto black_body = allocator.create<BlackbodySpectrum>(temperature);
    return allocator.create<Spectrum>(black_body);
}

const Spectrum *Spectrum::create_cie_d(Real temperature, const Real *cie_s0, const Real *cie_s1,
                                       const Real *cie_s2, const Real *cie_lambda,
                                       GPUMemoryAllocator &allocator) {
    Real cct = temperature * 1.4388f / 1.4380f;
    if (cct < 4000) {
        // CIE D ill-defined, use blackbody
        auto bb = BlackbodySpectrum(cct);

        auto densely_sampled_spectrum = allocator.create<DenselySampledSpectrum>();
        densely_sampled_spectrum->init_with_sample_function(
            [=](Real lambda) { return bb(lambda); });

        return allocator.create<Spectrum>(densely_sampled_spectrum);
    }

    // Convert CCT to xy
    Real x = cct <= 7000 ? -4.607f * 1e9f / pbrt::pow<3>(cct) + 2.9678f * 1e6f / sqr(cct) +
                               0.09911f * 1e3f / cct + 0.244063f
                         : -2.0064f * 1e9f / pbrt::pow<3>(cct) + 1.9018f * 1e6f / sqr(cct) +
                               0.24748f * 1e3f / cct + 0.23704f;

    Real y = -3 * x * x + 2.870f * x - 0.275f;

    // Interpolate D spectrum
    Real M = 0.0241f + 0.2562f * x - 0.7341f * y;
    Real M1 = (-1.3515f - 1.7703f * x + 5.9114f * y) / M;
    Real M2 = (0.0300f - 31.4424f * x + 30.0717f * y) / M;

    std::vector<Real> cpu_cie_lambdas(nCIES);
    std::vector<Real> cpu_values(nCIES);

    for (int idx = 0; idx < nCIES; ++idx) {
        cpu_cie_lambdas.push_back(cie_lambda[idx]);
        cpu_values.push_back((cie_s0[idx] + cie_s1[idx] * M1 + cie_s2[idx] * M2) * 0.01);
    }

    return create_piecewise_linear_spectrum_from_lambdas_and_values(cpu_cie_lambdas, cpu_values,
                                                                    allocator);
}

const Spectrum *Spectrum::create_constant_spectrum(Real val, GPUMemoryAllocator &allocator) {
    auto constant_spectrum = allocator.create<ConstantSpectrum>(val);

    return allocator.create<Spectrum>(constant_spectrum);
}

const Spectrum *Spectrum::create_from_rgb(const RGB &val, SpectrumType spectrum_type,
                                          const RGBColorSpace *color_space,
                                          GPUMemoryAllocator &allocator) {
    switch (spectrum_type) {
    case SpectrumType::Albedo: {
        if (val.r > 1 || val.g > 1 || val.b > 1) {
            REPORT_FATAL_ERROR();
        }

        auto rgb_albedo_spectrum = allocator.create<RGBAlbedoSpectrum>(val, color_space);
        return allocator.create<Spectrum>(rgb_albedo_spectrum);
    }

    case SpectrumType::Illuminant: {
        auto rgb_illuminant_spectrum = allocator.create<RGBIlluminantSpectrum>(val, color_space);
        return allocator.create<Spectrum>(rgb_illuminant_spectrum);
    }

    case SpectrumType::Unbounded: {
        auto rgb_unbounded_spectrum = allocator.create<RGBUnboundedSpectrum>(val, color_space);
        return allocator.create<Spectrum>(rgb_unbounded_spectrum);
    }
    }

    printf("\n SpectrumType `%d` not implemented\n\n", spectrum_type);

    REPORT_FATAL_ERROR();
    return nullptr;
}

const Spectrum *Spectrum::create_piecewise_linear_spectrum_from_lambdas_and_values(
    const std::vector<Real> &cpu_lambdas, const std::vector<Real> &cpu_values,
    GPUMemoryAllocator &allocator) {
    const auto piecewise_linear_spectrum =
        allocator.create<PiecewiseLinearSpectrum>(cpu_lambdas, cpu_values, allocator);
    return allocator.create<Spectrum>(piecewise_linear_spectrum);
}

const Spectrum *
Spectrum::create_piecewise_linear_spectrum_from_interleaved(const std::vector<Real> &samples,
                                                            bool normalize, const Spectrum *cie_y,
                                                            GPUMemoryAllocator &allocator) {
    const auto piecewise_linear_spectrum =
        allocator.create<PiecewiseLinearSpectrum>(samples, normalize, cie_y, allocator);
    return allocator.create<Spectrum>(piecewise_linear_spectrum);
}

PBRT_CPU_GPU
Real Spectrum::max_value() const {
    switch (type) {
    case Type::rgb_illuminant: {
        return static_cast<const RGBIlluminantSpectrum *>(ptr)->max_value();
    }
    }

    printf("type %d not implemented\n", type);
    REPORT_FATAL_ERROR();
    return NAN;
}

PBRT_CPU_GPU
Real Spectrum::operator()(Real lambda) const {
    switch (type) {
    case Type::black_body: {
        return static_cast<const BlackbodySpectrum *>(ptr)->operator()(lambda);
    }

    case Type::constant: {
        return static_cast<const ConstantSpectrum *>(ptr)->operator()(lambda);
    }

    case Type::densely_sampled: {
        return static_cast<const DenselySampledSpectrum *>(ptr)->operator()(lambda);
    }

    case Type::piecewise_linear: {
        return static_cast<const PiecewiseLinearSpectrum *>(ptr)->operator()(lambda);
    }

    case Type::rgb_albedo: {
        return static_cast<const RGBAlbedoSpectrum *>(ptr)->operator()(lambda);
    }

    case Type::rgb_illuminant: {
        return static_cast<const RGBIlluminantSpectrum *>(ptr)->operator()(lambda);
    }

    case Type::rgb_unbounded: {
        return static_cast<const RGBUnboundedSpectrum *>(ptr)->operator()(lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

PBRT_CPU_GPU
Real Spectrum::to_photometric(const Spectrum *cie_y) const {
    if (type == Type::rgb_illuminant) {
        return static_cast<const RGBIlluminantSpectrum *>(ptr)->to_photometric(cie_y);
    }

    return inner_product(cie_y);
}

PBRT_CPU_GPU
SampledSpectrum Spectrum::sample(const SampledWavelengths &lambda) const {
    switch (type) {
    case (Type::black_body): {
        return static_cast<const BlackbodySpectrum *>(ptr)->sample(lambda);
    }

    case (Type::constant): {
        return static_cast<const ConstantSpectrum *>(ptr)->sample(lambda);
    }

    case (Type::densely_sampled): {
        return static_cast<const DenselySampledSpectrum *>(ptr)->sample(lambda);
    }

    case (Type::piecewise_linear): {
        return static_cast<const PiecewiseLinearSpectrum *>(ptr)->sample(lambda);
    }

    case (Type::rgb_albedo): {
        return static_cast<const RGBAlbedoSpectrum *>(ptr)->sample(lambda);
    }

    case (Type::rgb_illuminant): {
        return static_cast<const RGBIlluminantSpectrum *>(ptr)->sample(lambda);
    }

    case (Type::rgb_unbounded): {
        return static_cast<const RGBUnboundedSpectrum *>(ptr)->sample(lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return SampledSpectrum(NAN);
}

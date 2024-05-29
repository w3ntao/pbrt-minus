#include "pbrt/base/spectrum.h"

#include "pbrt/spectra/densely_sampled_spectrum.h"
#include "pbrt/spectra/constant_spectrum.h"
#include "pbrt/spectra/rgb_illuminant_spectrum.h"
#include "pbrt/spectra/rgb_albedo_spectrum.h"

const Spectrum *Spectrum::create_constant_spectrum(FloatType val,
                                                   std::vector<void *> &gpu_dynamic_pointers) {
    ConstantSpectrum *constant_spectrum;
    Spectrum *spectrum;
    CHECK_CUDA_ERROR(cudaMallocManaged(&constant_spectrum, sizeof(ConstantSpectrum)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum, sizeof(Spectrum)));

    constant_spectrum->init(val);
    spectrum->init(constant_spectrum);

    gpu_dynamic_pointers.push_back(constant_spectrum);
    gpu_dynamic_pointers.push_back(spectrum);

    return spectrum;
}

const Spectrum *Spectrum::create_rgb_albedo_spectrum(const RGB &val,
                                                     std::vector<void *> &gpu_dynamic_pointers,
                                                     const RGBColorSpace *color_space) {
    RGBAlbedoSpectrum *rgb_albedo_spectrum;
    Spectrum *spectrum;

    CHECK_CUDA_ERROR(cudaMallocManaged(&rgb_albedo_spectrum, sizeof(RGBAlbedoSpectrum)));
    CHECK_CUDA_ERROR(cudaMallocManaged(&spectrum, sizeof(Spectrum)));

    rgb_albedo_spectrum->init(color_space, val);
    spectrum->init(rgb_albedo_spectrum);

    gpu_dynamic_pointers.push_back(rgb_albedo_spectrum);
    gpu_dynamic_pointers.push_back(spectrum);

    return spectrum;
}

PBRT_CPU_GPU
void Spectrum::init(const DenselySampledSpectrum *densely_sampled_spectrum) {
    type = Type::densely_sampled_spectrum;
    ptr = densely_sampled_spectrum;
}

PBRT_CPU_GPU
void Spectrum::init(const ConstantSpectrum *constant_spectrum) {
    type = Type::constant_spectrum;
    ptr = constant_spectrum;
}

PBRT_CPU_GPU
void Spectrum::init(const RGBIlluminantSpectrum *rgb_illuminant_spectrum) {
    type = Type::rgb_illuminant_spectrum;
    ptr = rgb_illuminant_spectrum;
}

PBRT_CPU_GPU
void Spectrum::init(const RGBAlbedoSpectrum *rgb_albedo_spectrum) {
    type = Type::rgb_albedo_spectrum;
    ptr = rgb_albedo_spectrum;
}

PBRT_CPU_GPU
FloatType Spectrum::operator()(FloatType lambda) const {
    switch (type) {
    case (Type::densely_sampled_spectrum): {
        return ((DenselySampledSpectrum *)ptr)->operator()(lambda);
    }

    case (Type::constant_spectrum): {
        return ((ConstantSpectrum *)ptr)->operator()(lambda);
    }

    case (Type::rgb_illuminant_spectrum): {
        return ((RGBIlluminantSpectrum *)ptr)->operator()(lambda);
    }

    case (Type::rgb_albedo_spectrum): {
        return ((RGBAlbedoSpectrum *)ptr)->operator()(lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return NAN;
}

PBRT_CPU_GPU
SampledSpectrum Spectrum::sample(const SampledWavelengths &lambda) const {
    switch (type) {
    case (Type::densely_sampled_spectrum): {
        return ((DenselySampledSpectrum *)ptr)->sample(lambda);
    }

    case (Type::constant_spectrum): {
        return ((ConstantSpectrum *)ptr)->sample(lambda);
    }

    case (Type::rgb_illuminant_spectrum): {
        return ((RGBIlluminantSpectrum *)ptr)->sample(lambda);
    }

    case (Type::rgb_albedo_spectrum): {
        return ((RGBAlbedoSpectrum *)ptr)->sample(lambda);
    }
    }

    REPORT_FATAL_ERROR();
    return SampledSpectrum(NAN);
}

#include <pbrt/base/spectrum.h>
#include <pbrt/spectrum_util/sampled_spectrum.h>
#include <pbrt/spectrum_util/sampled_wavelengths.h>

PBRT_CPU_GPU
Real SampledSpectrum::y(const SampledWavelengths &lambda, const Spectrum *cie_y) const {
    auto ys = cie_y->sample(lambda);
    auto pdf = lambda.pdf_as_sampled_spectrum();

    return (*this * ys).safe_div(pdf).average() / CIE_Y_integral;
}

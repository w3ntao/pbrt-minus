#pragma once

#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectra/rgb_albedo_spectrum.h>
#include <pbrt/spectrum_util/global_spectra.h>
#include <pbrt/spectrum_util/rgb_color_space.h>

class GPUMemoryAllocator;
class ParameterDictionary;
struct IntegratorBase;

class SurfaceNormalIntegrator {
  public:
    SurfaceNormalIntegrator(const ParameterDictionary &parameters, const IntegratorBase *_base)
        : base(_base) {
        const auto rgb_color_space = parameters.global_spectra->rgb_color_space;

        constexpr auto scale = 0.01;
        rgb_spectra[0] = RGBAlbedoSpectrum(RGB(scale, 0, 0), rgb_color_space);
        rgb_spectra[1] = RGBAlbedoSpectrum(RGB(0, scale, 0), rgb_color_space);
        rgb_spectra[2] = RGBAlbedoSpectrum(RGB(0, 0, scale), rgb_color_space);
    }

    PBRT_CPU_GPU
    SampledSpectrum li(const Ray &ray, const SampledWavelengths &lambda) const;

  private:
    const IntegratorBase *base = nullptr;

    RGBAlbedoSpectrum rgb_spectra[3];
};

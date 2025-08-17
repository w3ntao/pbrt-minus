#pragma once

#include <pbrt/euclidean_space/squared_matrix.h>
#include <pbrt/spectrum_util/rgb.h>
#include <pbrt/spectrum_util/rgb_color_space.h>

class PixelSensor {
  public:
    PixelSensor(const Spectrum *_r_bar, const Spectrum *_g_bar, const Spectrum *_b_bar,
                const Real _imaging_ratio, const SquareMatrix<3> &_xyz_from_sensor_rgb)
        : r_bar(_r_bar), g_bar(_g_bar), b_bar(_b_bar), imaging_ratio(_imaging_ratio),
          xyz_from_sensor_rgb(_xyz_from_sensor_rgb) {}

    static const PixelSensor *create_cie_1931(const Spectrum *const cie_xyz[3],
                                              const RGBColorSpace *output_color_space,
                                              const Spectrum *sensor_illum, Real imaging_ratio,
                                              GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    RGB to_sensor_rgb(const SampledSpectrum &radiance_l, const SampledWavelengths &lambda) const {
        const auto spectrum = radiance_l.safe_div(lambda.pdf_as_sampled_spectrum());

        return imaging_ratio * RGB((r_bar->sample(lambda) * spectrum).average(),
                                   (g_bar->sample(lambda) * spectrum).average(),
                                   (b_bar->sample(lambda) * spectrum).average());
    }

    SquareMatrix<3> xyz_from_sensor_rgb;

  private:
    const Spectrum *r_bar = nullptr;
    const Spectrum *g_bar = nullptr;
    const Spectrum *b_bar = nullptr;

    Real imaging_ratio = NAN;
};

#pragma once

#include "pbrt/base/spectrum.h"
#include "pbrt/euclidean_space/squared_matrix.h"
#include "pbrt/spectra/rgb.h"
#include "pbrt/spectra/rgb_color_space.h"
#include "pbrt/spectra/color_encoding.h"

class PixelSensor {
  public:
    PBRT_GPU PixelSensor()
        : r_bar(nullptr), g_bar(nullptr), b_bar(nullptr), imaging_ratio(0),
          xyz_from_sensor_rgb(SquareMatrix<3>()) {}

    PBRT_GPU PixelSensor(const Spectrum *_r_bar, const Spectrum *_g_bar, const Spectrum *_b_bar,
                         double _imaging_ratio, SquareMatrix<3> _xyz_from_sensor_rgb)
        : r_bar(_r_bar), g_bar(_g_bar), b_bar(_b_bar), imaging_ratio(_imaging_ratio),
          xyz_from_sensor_rgb(_xyz_from_sensor_rgb) {}

    PBRT_GPU static PixelSensor cie_1931(const std::array<const Spectrum *, 3> &cie_xyz,
                                         const RGBColorSpace *output_color_space,
                                         const Spectrum *sensor_illum, double imaging_ratio) {
        auto xyz_from_sensor_rgb = SquareMatrix<3>::identity();
        if (sensor_illum) {
            auto source_white = sensor_illum->to_xyz(cie_xyz).xy();
            auto target_white = output_color_space->w;

            xyz_from_sensor_rgb = white_balance(source_white, target_white);
        }

        return PixelSensor(cie_xyz[0], cie_xyz[1], cie_xyz[2], imaging_ratio, xyz_from_sensor_rgb);
    }

    PBRT_GPU RGB to_sensor_rgb(const SampledSpectrum &radiance_l,
                               const SampledWavelengths &lambda) const {
        const auto spectrum = radiance_l.safe_div(lambda.pdf_as_sampled_spectrum());

        return imaging_ratio * RGB((r_bar->sample(lambda) * spectrum).average(),
                                   (g_bar->sample(lambda) * spectrum).average(),
                                   (b_bar->sample(lambda) * spectrum).average());
    }

    SquareMatrix<3> xyz_from_sensor_rgb;

  private:
    const Spectrum *r_bar;
    const Spectrum *g_bar;
    const Spectrum *b_bar;

    double imaging_ratio;
};

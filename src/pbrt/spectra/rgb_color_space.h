#pragma once

#include "pbrt/util/macro.h"
#include "pbrt/base/spectrum.h"
#include "pbrt/spectra/rgb.h"
#include "pbrt/spectra/rgb_sigmoid_polynomial.h"
#include "pbrt/spectra/rgb_to_spectrum_data.h"

class RGBColorSpace {
  public:
    PBRT_CPU_GPU
    void init(const Point2f _r, const Point2f _g, const Point2f _b, const Spectrum *_illuminant,
              const RGBtoSpectrumData::RGBtoSpectrumTable *_rgb_to_spectrum_table,
              const Spectrum *cie_xyz[3]) {
        r = _r;
        g = _g;
        b = _b;
        illuminant = _illuminant;
        rgb_to_spectrum_table = _rgb_to_spectrum_table;

        // Compute whitepoint primaries and XYZ coordinates
        auto _whitepoint = illuminant->to_xyz(cie_xyz);

        w = _whitepoint.xy();

        auto xyz_r = XYZ::from_xyY(_r);
        auto xyz_g = XYZ::from_xyY(_g);
        auto xyz_b = XYZ::from_xyY(_b);

        FloatType array[3][3] = {
            {xyz_r.x, xyz_g.x, xyz_b.x},
            {xyz_r.y, xyz_g.y, xyz_b.y},
            {xyz_r.z, xyz_g.z, xyz_b.z},
        };

        auto rgb = SquareMatrix<3>(array);

        // Initialize XYZ color space conversion matrices
        XYZ c = rgb.inverse() * _whitepoint;
        FloatType diagonal_data[3] = {c[0], c[1], c[2]};
        xyz_from_rgb = rgb * SquareMatrix<3>::diag(diagonal_data);
        rgb_from_xyz = xyz_from_rgb.inverse();
    }

    PBRT_CPU_GPU
    RGBSigmoidPolynomial to_rgb_coefficients(const RGB &rgb) const {
        return (*rgb_to_spectrum_table)(rgb.clamp_zero());
    }

    // RGBColorSpace Public Members
    Point2f r;
    Point2f g;
    Point2f b;
    Point2f w;

    const Spectrum *illuminant;
    SquareMatrix<3> xyz_from_rgb;
    SquareMatrix<3> rgb_from_xyz;
    const RGBtoSpectrumData::RGBtoSpectrumTable *rgb_to_spectrum_table;
};

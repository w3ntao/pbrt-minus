#pragma once

#include "pbrt/util/macro.h"
#include "pbrt/spectra/spectrum.h"
#include "pbrt/spectra/densely_sampled_spectrum.h"
#include "pbrt/spectra/rgb_to_spectrum_data.h"

class RGBColorSpace {
  public:
    PBRT_GPU RGBColorSpace(const Point2f _r, const Point2f _g, const Point2f _b,
                           const Spectrum &_illuminant,
                           const RGBtoSpectrumData::RGBtoSpectrumTableGPU *_rgb_to_spectrum_table,
                           const std::array<const Spectrum *, 3> &cie_xyz)
        : r(_r), g(_g), b(_b), illuminant(new DenselySampledSpectrum(_illuminant)),
          rgb_to_spectrum_table(_rgb_to_spectrum_table) {

        // Compute whitepoint primaries and XYZ coordinates
        auto whitepoint = illuminant->to_xyz(cie_xyz);

        auto xyz_r = XYZ::from_xyY(_r);
        auto xyz_g = XYZ::from_xyY(_g);
        auto xyz_b = XYZ::from_xyY(_b);

        double array[3][3] = {
            {xyz_r.x, xyz_g.x, xyz_b.x},
            {xyz_r.y, xyz_g.y, xyz_b.y},
            {xyz_r.z, xyz_g.z, xyz_b.z},
        };
        auto rgb = SquareMatrix<3>(array);

        // Initialize XYZ color space conversion matrices
        XYZ c = rgb.inverse() * whitepoint;
        double diag_data[3] = {c[0], c[1], c[2]};
        XYZFromRGB = rgb * SquareMatrix<3>::diag(diag_data);
        RGBFromXYZ = XYZFromRGB.inverse();
    }

    PBRT_GPU ~RGBColorSpace() {
        delete illuminant;
    }

    /*
    PBRT_CPU_GPU
    RGBSigmoidPolynomial ToRGBCoeffs(RGB rgb) const;
    */

    // RGBColorSpace Public Members
    Point2f r;
    Point2f g;
    Point2f b;
    Point2f w;

    const DenselySampledSpectrum *illuminant;
    SquareMatrix<3> XYZFromRGB;
    SquareMatrix<3> RGBFromXYZ;
    const RGBtoSpectrumData::RGBtoSpectrumTableGPU *rgb_to_spectrum_table;

    /*
    PBRT_CPU_GPU
    bool operator==(const RGBColorSpace &cs) const {
        return (r == cs.r && g == cs.g && b == cs.b && w == cs.w &&
                rgbToSpectrumTable == cs.rgbToSpectrumTable);
    }
    PBRT_CPU_GPU
    bool operator!=(const RGBColorSpace &cs) const {
        return (r != cs.r || g != cs.g || b != cs.b || w != cs.w ||
                rgbToSpectrumTable != cs.rgbToSpectrumTable);
    }

    PBRT_CPU_GPU
    RGB LuminanceVector() const {
        return RGB(XYZFromRGB[1][0], XYZFromRGB[1][1], XYZFromRGB[1][2]);
    }

    PBRT_CPU_GPU
    RGB ToRGB(XYZ xyz) const {
        return Mul<RGB>(RGBFromXYZ, xyz);
    }
    PBRT_CPU_GPU
    XYZ ToXYZ(RGB rgb) const {
        return Mul<XYZ>(XYZFromRGB, rgb);
    }
    */
};

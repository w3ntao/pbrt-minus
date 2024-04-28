#include "pbrt/base/spectrum.h"
#include "pbrt/spectrum_util/rgb_to_spectrum_data.h"
#include "pbrt/spectrum_util/rgb_color_space.h"

namespace GPU {

struct GlobalVariable {
    void init(const Spectrum *_cie_xyz[3], const Spectrum *cie_illum_d6500,
              const RGBtoSpectrumData::RGBtoSpectrumTable *rgb_to_spectrum_table,
              RGBtoSpectrumData::Gamut gamut) {
        for (uint idx = 0; idx < 3; idx++) {
            cie_xyz[idx] = _cie_xyz[idx];
        }

        if (gamut == RGBtoSpectrumData::Gamut::sRGB) {
            rgb_color_space->init(Point2f(0.64, 0.33), Point2f(0.3, 0.6), Point2f(0.15, 0.06),
                                  cie_illum_d6500, rgb_to_spectrum_table, cie_xyz);

            return;
        }

        throw std::runtime_error(
            "\nGlobalVariable::init(): this color space is not implemented\n\n");
    }

    PBRT_CPU_GPU void get_cie_xyz(const Spectrum *out[3]) const {
        for (uint idx = 0; idx < 3; idx++) {
            out[idx] = cie_xyz[idx];
        }
    }

    RGBColorSpace *rgb_color_space;
    const Spectrum *cie_xyz[3];
};
} // namespace GPU

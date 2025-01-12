#pragma once

#include "pbrt/base/spectrum.h"
#include "pbrt/spectrum_util/rgb_to_spectrum_data.h"

class Spectrum;

struct GlobalSpectra {
    static const GlobalSpectra *create(RGBtoSpectrumData::Gamut gamut,
                                       std::vector<void *> &gpu_dynamic_pointers);
    const RGBColorSpace *rgb_color_space;
    const Spectrum *cie_xyz[3];
    const Spectrum *cie_y;
};

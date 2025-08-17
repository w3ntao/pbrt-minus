#pragma once

#include <pbrt/base/spectrum.h>
#include <pbrt/spectrum_util/rgb_to_spectrum_data.h>

class GPUMemoryAllocator;
class Spectrum;

struct GlobalSpectra {
    GlobalSpectra(const RGBColorSpace *_rgb_color_space, const Spectrum *_cie_xyz[3])
        : rgb_color_space(_rgb_color_space) {
        for (int idx = 0; idx < 3; ++idx) {
            cie_xyz[idx] = _cie_xyz[idx];
        }
        cie_y = _cie_xyz[1];
    }

    static const GlobalSpectra *create(RGBtoSpectrumData::Gamut gamut,
                                       GPUMemoryAllocator &allocator);

    const RGBColorSpace *rgb_color_space = nullptr;
    const Spectrum *cie_xyz[3] = {nullptr, nullptr, nullptr};
    const Spectrum *cie_y = nullptr;
};

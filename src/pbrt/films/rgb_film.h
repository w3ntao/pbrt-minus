#pragma once

#include "pbrt/spectra/rgb.h"

struct Pixel {
    RGB rgb_sum;
    double weight_sum;

    PBRT_GPU Pixel() : rgb_sum(RGB(0.0, 0.0, 0.0)), weight_sum(0.0) {}
};

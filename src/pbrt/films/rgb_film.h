#pragma once

struct Pixel {
    double rgb_sum[3];
    // TODO: replace double[3] with RGB
    double weight_sum;

    PBRT_GPU Pixel() : rgb_sum{0, 0, 0}, weight_sum(0) {}
};

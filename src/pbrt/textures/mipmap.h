#pragma once

#include <vector>

#include "pbrt/util/macro.h"
#include "pbrt/textures/gpu_image.h"

class RGBColorSpace;
class GPUImage;

enum class FilterFunction {
    Point,
    Bilinear,
    Trilinear,
    EWA,
};

static FilterFunction parse_filter_function(const std::string &filter_function) {
    if (filter_function == "ewa") {
        return FilterFunction::EWA;
    }

    if (filter_function == "bilinear") {
        return FilterFunction::Bilinear;
    }

    if (filter_function == "trilinear") {
        return FilterFunction::Trilinear;
    }

    if (filter_function == "point") {
        return FilterFunction::Point;
    }

    REPORT_FATAL_ERROR();
    return FilterFunction::EWA;
}

struct MIPMapFilterOptions {
    FilterFunction filter;
    // default value: FilterFunction::EWA
    FloatType max_anisotropy;
    // default value: 8.0
};

class MIPMap {
  public:
    void init(const MIPMapFilterOptions &_options, WrapMode _wrap_mode, const std::string &filepath,
              const RGBColorSpace *_color_space, std::vector<void *> &gpu_dynamic_pointers) {
        options = _options;
        wrap_mode = _wrap_mode;
        color_space = _color_space;

        GPUImage *gpu_image;
        CHECK_CUDA_ERROR(cudaMallocManaged(&gpu_image, sizeof(GPUImage)));

        gpu_image->init(filepath, gpu_dynamic_pointers);
        image = gpu_image;

        gpu_dynamic_pointers.push_back(gpu_image);
    }

    PBRT_CPU_GPU
    RGB filter(const Point2f st) const {
        return image->bilerp(st, WrapMode2D(wrap_mode));
    }

  private:
    GPUImage *image;
    const RGBColorSpace *color_space;
    WrapMode wrap_mode;
    MIPMapFilterOptions options;
};

#pragma once

#include <pbrt/euclidean_space/point2.h>

enum class WrapMode;
class GPUImage;
class GPUMemoryAllocator;
class ParameterDictionary;
class RGB;
class RGBColorSpace;

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
    Real max_anisotropy = NAN;
    // default value: 8.0
};

class MIPMap {
  public:
    MIPMap(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    RGB filter(const Point2f st) const;

  private:
    const GPUImage *image = nullptr;
    WrapMode wrap_mode;
    MIPMapFilterOptions options;
};

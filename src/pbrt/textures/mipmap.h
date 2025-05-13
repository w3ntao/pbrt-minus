#pragma once

#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/spectrum_util/rgb.h>
#include <pbrt/textures/gpu_image.h>
#include <pbrt/gpu/macro.h>

class GPUImage;
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
    Real max_anisotropy;
    // default value: 8.0
};

class MIPMap {
  public:
    static const MIPMap *create(const ParameterDictionary &parameters,
                                GPUMemoryAllocator &allocator) {
        auto mipmap = allocator.allocate<MIPMap>();
        mipmap->init(parameters, allocator);

        return mipmap;
    }

    PBRT_CPU_GPU
    RGB filter(const Point2f st) const {
        return image->bilerp(st, WrapMode2D(wrap_mode));
    }

  private:
    const GPUImage *image;
    WrapMode wrap_mode;
    MIPMapFilterOptions options;

    void init(const ParameterDictionary &parameters, GPUMemoryAllocator &allocator) {
        auto max_anisotropy = parameters.get_float("maxanisotropy", 8.0);
        auto filter_string = parameters.get_one_string("filter", "bilinear");

        options = MIPMapFilterOptions{
            .filter = parse_filter_function(filter_string),
            .max_anisotropy = max_anisotropy,
        };

        auto wrap_string = parameters.get_one_string("wrap", "repeat");
        wrap_mode = parse_wrap_mode(wrap_string);

        auto image_path = parameters.root + "/" + parameters.get_one_string("filename");
        image = GPUImage::create_from_file(image_path, allocator);
    }
};

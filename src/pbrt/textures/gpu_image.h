#pragma once

#include <string>
#include <vector>

#include "pbrt/util/macro.h"
#include "pbrt/euclidean_space/point2.h"

class RGB;

enum class WrapMode {
    Black,
    Clamp,
    Repeat,
    OctahedralSphere,
};

static WrapMode parse_wrap_mode(const std::string &wrap_mode) {
    if (wrap_mode == "black") {
        return WrapMode::Black;
    }

    if (wrap_mode == "clamp") {
        return WrapMode::Clamp;
    }

    if (wrap_mode == "repeat") {
        return WrapMode::Repeat;
    }

    if (wrap_mode == "octahedralsphere") {
        return WrapMode::OctahedralSphere;
    }

    REPORT_FATAL_ERROR();
    return WrapMode::Black;
}

struct WrapMode2D {
    PBRT_CPU_GPU
    WrapMode2D(WrapMode w) : wrap{w, w} {}

    PBRT_CPU_GPU
    WrapMode2D(WrapMode x, WrapMode y) : wrap{x, y} {}

    PBRT_CPU_GPU
    WrapMode operator[](const uint idx) const {
        return wrap[idx];
    }

    WrapMode wrap[2];
};

enum class PixelFormat {
    U256,
    Half,
    Float,
};

class GPUImage {
  public:
    void init(const std::string &filename, std::vector<void *> &gpu_dynamic_pointers);

    PBRT_CPU_GPU
    RGB fetch_pixel(Point2i _p, WrapMode2D wrap_mode) const;

    PBRT_CPU_GPU RGB bilerp(Point2f p, WrapMode2D wrap) const;

  private:
    const RGB *pixels;
    Point2i resolution;
    PixelFormat pixel_format;
};

#pragma once

#include <pbrt/euclidean_space/point2.h>
#include <pbrt/gpu/macro.h>
#include <string>

class GPUMemoryAllocator;
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
    WrapMode operator[](const int idx) const {
        return wrap[idx];
    }

    WrapMode wrap[2];
};

enum class PixelFormat {
    U256,
    Half,
    Float,
};

// PixelFormat Inline Functions
PBRT_CPU_GPU
inline bool Is8Bit(PixelFormat format) {
    return format == PixelFormat::U256;
}

PBRT_CPU_GPU
inline bool Is16Bit(PixelFormat format) {
    return format == PixelFormat::Half;
}

PBRT_CPU_GPU
inline bool Is32Bit(PixelFormat format) {
    return format == PixelFormat::Float;
}

class GPUImage {
  public:
    static const GPUImage *create_from_file(const std::string &filename,
                                            GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Point2i get_resolution() const {
        return resolution;
    }

    PBRT_CPU_GPU
    RGB fetch_pixel(Point2i _p, WrapMode2D wrap_mode) const;

    PBRT_CPU_GPU RGB bilerp(Point2f p, WrapMode2D wrap) const;

  private:
    Point2i resolution;

    const RGB *pixels = nullptr;

    void init_exr(const std::string &filename, GPUMemoryAllocator &allocator);

    void init_pfm(const std::string &filename, GPUMemoryAllocator &allocator);

    void init_png(const std::string &filename, GPUMemoryAllocator &allocator);

    void init_tga(const std::string &filename, GPUMemoryAllocator &allocator);
};

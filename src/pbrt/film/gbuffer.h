#pragma once

#include <pbrt/euclidean_space/point2.h>
#include <pbrt/euclidean_space/vector3.h>
#include <pbrt/spectrum_util/rgb.h>
#include <string>

class GPUMemoryAllocator;

struct GBufferPixel {
    Real ambient_occlusion = 0;
    Vector3f normal{0, 0, 0};
    RGB albedo{0};
    Real depth = Infinity;
    Real coverage = 0;
    // Linear scene luminance and its maximum absolute 3x3 neighborhood
    // difference, populated from a rendered radiance image when available.
    Real luminance = NAN;
    Real luminance_difference = NAN;
};

enum class GBufferChannel {
    AmbientOcclusion,
    SurfaceNormal,
    Albedo,
    Depth,
    Luminance,
    LuminanceDifference,
};

class GBuffer {
  public:
    GBuffer(Point2i resolution, GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    GBufferPixel *data() {
        return pixels;
    }

    PBRT_CPU_GPU
    const GBufferPixel *data() const {
        return pixels;
    }

    PBRT_CPU_GPU
    Point2i get_resolution() const {
        return resolution;
    }

    // GPU writes must be complete before reading the managed pixels on the CPU.
    void write_to_exr(const std::string &filename) const;
    void write_to_png(const std::string &filename, GBufferChannel channel) const;
    void copy_to_frame_buffer(uint8_t *frame_buffer, GBufferChannel channel) const;

  private:
    Point2i resolution;
    GBufferPixel *pixels;
};

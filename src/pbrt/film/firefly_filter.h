#pragma once

#include <cstddef>
#include <pbrt/euclidean_space/point2.h>
#include <pbrt/spectrum_util/rgb.h>

class GPUMemoryAllocator;
struct GBufferPixel;

namespace statmc {
struct TransformedStatistics;
}

namespace firefly {

// A conservative spatial prefilter for the completed HDR image, including
// normalized splats. This is not the paper's per-camera-sample reweighting:
// small isolated real highlights can also be attenuated.
// The allocator owns the buffers and must outlive this object and its users.
class Prefilter {
  public:
    Prefilter(Point2i resolution, GPUMemoryAllocator &allocator);

    // Input and guides must be GPU-accessible and ready to read. Guide
    // luminance must come from this input via set_luminance_guide(), so its
    // color-space conversion matches the renderer. Rebuild that guide from
    // the returned image before calling StatMC.
    // Returns synchronized, separate output; input and guides are unchanged.
    // Optional Box-Cox moments are rescaled consistently with the RGB clamp.
    const RGB *filter(const RGB *linear_rgb, const GBufferPixel *guides,
                      statmc::TransformedStatistics *statistics = nullptr);

    // Available after filter(): a common RGB scale in [0, 1] for each pixel.
    const Real *scales() const {
        return pixel_scales;
    }

    size_t modified_pixels() const {
        return modified_pixel_count;
    }

  private:
    Point2i resolution;
    RGB *output;
    Real *pixel_scales;
    size_t modified_pixel_count = 0;
};

} // namespace firefly

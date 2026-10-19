#pragma once

#include <pbrt/euclidean_space/point2.h>
#include <pbrt/film/firefly_detector.h>
#include <string>

class GPUMemoryAllocator;
class GreyScaleFilm;

namespace firefly {

struct CascadeView {
    Real **counts;
    WeightRange **ranges;
    uint8_t *mask;

    // Calls for one pixel must have a serialized owner; different pixels may
    // be recorded concurrently. This keeps the per-sample updates atomics-free.
    PBRT_CPU_GPU
    void record(const int pixel_index, const Real luminance) const {
        const auto sample = split(luminance);
        if (sample.lower_layer < 0) {
            return;
        }

        // Every positive finite camera sample contributes one histogram count,
        // including samples that are forced outliers.
        counts[sample.lower_layer][pixel_index] += sample.lower_count;
        counts[sample.lower_layer + 1][pixel_index] += sample.upper_count;
        ranges[sample.lower_layer][pixel_index].record(sample.lower_count);
        if (sample.forced_outlier) {
            mask[pixel_index] = 1;
        }
    }
};

// Allocations belong to the supplied allocator, which must outlive this cascade
// and all kernels using its view.
class Cascade {
  public:
    Cascade(Point2i resolution, GPUMemoryAllocator &allocator);

    CascadeView view() const {
        return detector;
    }

    void finalize(GreyScaleFilm &firefly_mask, int samples_per_pixel, const std::string &output_filename) const;

  private:
    Point2i resolution;
    CascadeView detector;
};

} // namespace firefly

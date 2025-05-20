#pragma once

#include <cuda/std/variant>
#include <pbrt/filters/box.h>
#include <pbrt/filters/gaussian.h>
#include <pbrt/filters/mitchell.h>
#include <pbrt/filters/triangle.h>

namespace HIDDEN {
using FilterVariants =
    cuda::std::variant<BoxFilter, GaussianFilter, MitchellFilter, TriangleFilter>;
}

class Filter : public HIDDEN::FilterVariants {
    using HIDDEN::FilterVariants::FilterVariants;

  public:
    static const Filter *create(const std::string &filter_type,
                                const ParameterDictionary &parameters,
                                GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Vector2f get_radius() const {
        return cuda::std::visit([&](auto &x) { return x.get_radius(); }, *this);
    }

    PBRT_CPU_GPU
    Real get_integral() const {
        return cuda::std::visit([&](auto &x) { return x.get_integral(); }, *this);
    }

    PBRT_CPU_GPU
    Real evaluate(const Point2f &p) const {
        return cuda::std::visit([&](auto &x) { return x.evaluate(p); }, *this);
    }

    PBRT_CPU_GPU
    FilterSample sample(const Point2f &u) const;
};

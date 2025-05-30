#pragma once

#include <cuda/std/variant>
#include <pbrt/film/rgb_film.h>

namespace HIDDEN {
using FilmVariants = cuda::std::variant<RGBFilm>;
}

class Film : public HIDDEN::FilmVariants {
    using HIDDEN::FilmVariants::FilmVariants;

  public:
    static Film *create_rgb_film(const Filter *filter, const ParameterDictionary &parameters,
                                 GPUMemoryAllocator &allocator);

    PBRT_CPU_GPU
    Point2i get_resolution() const {
        return cuda::std::visit([&](auto &x) { return x.get_resolution(); }, *this);
    }

    PBRT_CPU_GPU
    const Filter *get_filter() const {
        return cuda::std::visit([&](auto &x) { return x.get_filter(); }, *this);
    }

    PBRT_CPU_GPU
    Bounds2f sample_bounds() const {
        return cuda::std::visit([&](auto &x) { return x.sample_bounds(); }, *this);
    }

    PBRT_CPU_GPU
    void add_sample(int pixel_index, const SampledSpectrum &radiance_l,
                    const SampledWavelengths &lambda, Real weight) {
        cuda::std::visit([&](auto &x) { x.add_sample(pixel_index, radiance_l, lambda, weight); },
                         *this);
    }

    PBRT_CPU_GPU
    void add_sample(const Point2i &p_film, const SampledSpectrum &radiance_l,
                    const SampledWavelengths &lambda, Real weight) {
        cuda::std::visit([&](auto &x) { x.add_sample(p_film, radiance_l, lambda, weight); }, *this);
    }

    // CPU only
    void add_splat(const Point2f &p_film, const SampledSpectrum &radiance_l,
                   const SampledWavelengths &lambda) {
        cuda::std::visit([&](auto &x) { x.add_splat(p_film, radiance_l, lambda); }, *this);
    }

    PBRT_CPU_GPU
    RGB get_pixel_rgb(const Point2i &p, Real splat_scale = 1) const {
        return cuda::std::visit([&](auto &x) { return x.get_pixel_rgb(p, splat_scale); }, *this);
    }

    // CPU only
    void copy_to_frame_buffer(uint8_t *gpu_frame_buffer, Real splat_scale = 1) const;

    // CPU only
    void write_to_png(const std::string &filename, Real splat_scale = 1) const;
};

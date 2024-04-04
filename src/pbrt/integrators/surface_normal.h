#pragma once

#include "pbrt/base/integrator.h"
#include "pbrt/spectra/rgb_color_space.h"
#include "pbrt/spectra/rgb_albedo_spectrum.h"
#include "pbrt/films/pixel_sensor.h"

class SurfaceNormalIntegrator {
  public:
    void init(const RGBColorSpace *rgb_color_space) {
        RGBAlbedoSpectrum::build_albedo_rgb(rgb_spectra, rgb_color_space);
    }

    PBRT_GPU SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda, const HLBVH *bvh,
                                Sampler &sampler) const {
        const auto shape_intersection = bvh->intersect(ray, Infinity);
        if (!shape_intersection) {
            return SampledSpectrum::same_value(0);
        }

        const Vector3f normal =
            shape_intersection->interaction.n.to_vector3().face_forward(-ray.d).normalize();

        const auto color = normal.softmax();

        return color[0] * rgb_spectra[0].sample(lambda) + color[1] * rgb_spectra[1].sample(lambda) +
               color[2] * rgb_spectra[2].sample(lambda);
    }

  private:
    RGBAlbedoSpectrum rgb_spectra[3];
};

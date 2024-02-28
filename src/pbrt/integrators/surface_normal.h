#pragma once

#include "pbrt/base/integrator.h"
#include "pbrt/spectra/rgb_color_space.h"
#include "pbrt/spectra/rgb_albedo_spectrum.h"
#include "pbrt/films/pixel_sensor.h"

class SurfaceNormalIntegrator : public Integrator {
  public:
    PBRT_GPU SurfaceNormalIntegrator(const RGBColorSpace &rgb_color_space,
                                     const PixelSensor &sensor)
        : rgb_spectra(RGBAlbedoSpectrum::build_albedo_rgb(rgb_color_space)) {}

    PBRT_GPU SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda,
                                const Aggregate *aggregate, Sampler &sampler) const override {
        const auto shape_intersection = aggregate->intersect(ray);
        if (!shape_intersection) {
            return SampledSpectrum(0);
        }

        const Vector3f normal =
            shape_intersection->interation.n.to_vector3().face_forward(-ray.d).normalize();

        const auto color = normal.softmax();

        return color[0] * rgb_spectra[0].sample(lambda) + color[1] * rgb_spectra[1].sample(lambda) +
               color[2] * rgb_spectra[2].sample(lambda);
    }

  private:
    std::array<RGBAlbedoSpectrum, 3> rgb_spectra;
};

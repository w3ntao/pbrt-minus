#pragma once

#include "pbrt/base/ray.h"
#include "pbrt/base/sampler.h"
#include "pbrt/base/spectrum.h"
#include "pbrt/accelerator/hlbvh.h"
#include "pbrt/euclidean_space/frame.h"
#include "pbrt/integrators/integrator_base.h"
#include "pbrt/spectrum_util/rgb_color_space.h"
#include "pbrt/spectra/rgb_albedo_spectrum.h"
#include "pbrt/util/sampling.h"

class SurfaceNormalIntegrator {
  public:
    void init(const IntegratorBase *_base, const RGBColorSpace *rgb_color_space) {
        base = _base;

        auto scale = 0.01;
        rgb_spectra[0].init(rgb_color_space, RGB(scale, 0, 0));
        rgb_spectra[1].init(rgb_color_space, RGB(0, scale, 0));
        rgb_spectra[2].init(rgb_color_space, RGB(0, 0, scale));
    }

    PBRT_GPU SampledSpectrum li(const DifferentialRay &ray, SampledWavelengths &lambda) const {
        const auto shape_intersection = base->bvh->intersect(ray.ray, Infinity);
        if (!shape_intersection) {
            return SampledSpectrum(0.0);
        }

        const Vector3f normal =
            shape_intersection->interaction.n.to_vector3().face_forward(-ray.ray.d).normalize();

        const auto color = normal.softmax();

        return color[0] * rgb_spectra[0].sample(lambda) + color[1] * rgb_spectra[1].sample(lambda) +
               color[2] * rgb_spectra[2].sample(lambda);
    }

  private:
    const IntegratorBase *base;

    RGBAlbedoSpectrum rgb_spectra[3];
};

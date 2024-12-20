#pragma once

#include "pbrt/accelerator/hlbvh.h"
#include "pbrt/base/integrator_base.h"
#include "pbrt/base/ray.h"
#include "pbrt/base/spectrum.h"
#include "pbrt/spectra/rgb_albedo_spectrum.h"
#include "pbrt/spectrum_util/rgb_color_space.h"
#include "pbrt/util/sampling.h"

class SurfaceNormalIntegrator {
  public:
    static const SurfaceNormalIntegrator *create(const ParameterDictionary &parameters,
                                                 const IntegratorBase *integrator_base,
                                                 std::vector<void *> &gpu_dynamic_pointers);

    void init(const IntegratorBase *_base, const RGBColorSpace *rgb_color_space) {
        base = _base;

        auto scale = 0.01;
        rgb_spectra[0].init(RGB(scale, 0, 0), rgb_color_space);
        rgb_spectra[1].init(RGB(0, scale, 0), rgb_color_space);
        rgb_spectra[2].init(RGB(0, 0, scale), rgb_color_space);
    }

    PBRT_GPU SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda) const {
        const auto shape_intersection = base->intersect(ray, Infinity);
        if (!shape_intersection) {
            return SampledSpectrum(0.0);
        }

        const Vector3f normal =
            shape_intersection->interaction.n.to_vector3().face_forward(-ray.d).normalize();

        const auto color = normal.softmax();

        return color[0] * rgb_spectra[0].sample(lambda) + color[1] * rgb_spectra[1].sample(lambda) +
               color[2] * rgb_spectra[2].sample(lambda);
    }

  private:
    const IntegratorBase *base;

    RGBAlbedoSpectrum rgb_spectra[3];
};

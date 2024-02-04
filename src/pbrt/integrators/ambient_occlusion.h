#pragma once

#include "pbrt/base/ray.h"
#include "pbrt/base/integrator.h"
#include "pbrt/util/sampling.h"
#include "pbrt/euclidean_space/frame.h"

class AmbientOcclusionIntegrator : public Integrator {
  public:
    PBRT_GPU AmbientOcclusionIntegrator(const Spectrum *_illuminant_spectrum,
                                        double _illuminant_scale)
        : illuminant_spectrum(_illuminant_spectrum), illuminant_scale(_illuminant_scale) {}

    PBRT_GPU SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda,
                                const Aggregate *aggregate, Sampler &sampler) const override {
        const auto shape_intersection = aggregate->intersect(ray);
        if (!shape_intersection) {
            return SampledSpectrum(0);
        }

        const SurfaceInteraction &isect = shape_intersection->interation;

        auto normal = isect.n.to_vector3().face_forward(-ray.d);

        auto u = sampler.get_2d();
        auto local_wi = sample_cosine_hemisphere(u);
        auto pdf = cosine_hemisphere_pdf(std::abs(local_wi.z));

        if (pdf == 0.0) {
            return SampledSpectrum(0);
        }

        auto frame = Frame::from_z(normal);
        auto wi = frame.from_local(local_wi);

        // Divide by PI so that fully visible is one.
        auto spawned_ray = isect.spawn_ray(wi);

        if (aggregate->fast_intersect(spawned_ray, Infinity)) {
            return SampledSpectrum(0);
        }

        return illuminant_spectrum->sample(lambda) *
               (illuminant_scale * normal.dot(wi) / (compute_pi() * pdf));
    }

  private:
    const Spectrum *illuminant_spectrum;
    double illuminant_scale;
};

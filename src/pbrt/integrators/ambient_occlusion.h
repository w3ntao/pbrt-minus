#pragma once

#include "pbrt/accelerator/hlbvh.h"

#include "pbrt/base/ray.h"
#include "pbrt/base/sampler.h"
#include "pbrt/base/spectrum.h"

#include "pbrt/euclidean_space/frame.h"
#include "pbrt/integrators/integrator_base.h"
#include "pbrt/util/sampling.h"

class IntegratorBase;
class ParameterDictionary;

class AmbientOcclusionIntegrator {
  public:
    static const AmbientOcclusionIntegrator *create(const ParameterDictionary &parameters,
                                                    const IntegratorBase *integrator_base,
                                                    std::vector<void *> &gpu_dynamic_pointers);

    void init(const IntegratorBase *_base, const Spectrum *_illuminant_spectrum,
              const FloatType _illuminant_scale) {
        base = _base;
        illuminant_spectrum = _illuminant_spectrum;
        illuminant_scale = _illuminant_scale;
    }

    PBRT_GPU SampledSpectrum li(const Ray &ray, SampledWavelengths &lambda,
                                Sampler *sampler) const {
        const auto shape_intersection = base->bvh->intersect(ray, Infinity);

        if (!shape_intersection) {
            return SampledSpectrum(0.0);
        }

        const SurfaceInteraction &isect = shape_intersection->interaction;

        auto normal = isect.n.to_vector3().face_forward(-ray.d);

        auto u = sampler->get_2d();
        auto local_wi = sample_cosine_hemisphere(u);
        auto pdf = cosine_hemisphere_pdf(std::abs(local_wi.z));

        if (pdf == 0.0) {
            return SampledSpectrum(0.0);
        }

        auto frame = Frame::from_z(normal);
        auto wi = frame.from_local(local_wi);

        // Divide by PI so that fully visible is one.
        auto spawned_ray = isect.spawn_ray(wi);

        if (base->bvh->fast_intersect(spawned_ray, Infinity)) {
            return SampledSpectrum(0.0);
        }

        return illuminant_spectrum->sample(lambda) *
               (illuminant_scale * normal.dot(wi) / (compute_pi() * pdf));
    }

  private:
    const IntegratorBase *base;
    const Spectrum *illuminant_spectrum;
    FloatType illuminant_scale;
};

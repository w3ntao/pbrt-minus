#include "pbrt/accelerator/hlbvh.h"
#include "pbrt/base/integrator_base.h"
#include "pbrt/base/interaction.h"

PBRT_GPU
bool IntegratorBase::fast_intersect(const Ray &ray, FloatType t_max) const {
    return bvh->fast_intersect(ray, t_max);
}

PBRT_GPU
bool IntegratorBase::unoccluded(const Interaction &p0, const Interaction &p1) const {
    return !fast_intersect(p0.spawn_ray_to(p1), 0.6) && !fast_intersect(p1.spawn_ray_to(p0), 0.6);
}

PBRT_GPU
pbrt::optional<ShapeIntersection> IntegratorBase::intersect(const Ray &ray,
                                                                 FloatType t_max) const {
    return bvh->intersect(ray, t_max);
}

PBRT_GPU
SampledSpectrum IntegratorBase::tr(const Interaction &p0, const Interaction &p1) const {
    auto ray = p0.spawn_ray_to(p1);

    SampledSpectrum Tr(1.f);
    SampledSpectrum inv_w(1.f);

    if (ray.d.squared_length() == 0) {
        return Tr;
    }

    while (true) {
        auto si = intersect(ray, 1 - ShadowEpsilon);
        // Handle opaque surface along ray's path

        if (si && si->interaction.material != nullptr) {
            return SampledSpectrum(0.0f);
        }

        // TODO: handle ray.medium here

        // Generate next ray segment or return final transmittance
        if (!si) {
            break;
        }

        ray = si->interaction.spawn_ray_to(p1);
    }

    return Tr / inv_w.average();
}

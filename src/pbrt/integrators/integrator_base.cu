#include "pbrt/integrators/integrator_base.h"

#include "pbrt/accelerator/hlbvh.h"
#include "pbrt/base/interaction.h"

PBRT_GPU
bool IntegratorBase::fast_intersect(const Ray &ray, FloatType t_max) const {
    return bvh->fast_intersect(ray, t_max);
}

PBRT_GPU
bool IntegratorBase::unoccluded(const Interaction &p0, const Interaction &p1) const {
    return !fast_intersect(p0.spawn_ray_to(p1), 0.6) && !fast_intersect(p1.spawn_ray_to(p0), 0.6);
}

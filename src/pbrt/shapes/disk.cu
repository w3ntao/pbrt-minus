#include <pbrt/base/shape.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/shapes/disk.h>
#include <pbrt/util/sampling.h>

const Disk *Disk::create(const Transform &render_from_object, const Transform &object_from_render,
                         bool reverse_orientation, const ParameterDictionary &parameters,
                         GPUMemoryAllocator &allocator) {
    auto disk = allocator.allocate<Disk>();

    disk->render_from_object = render_from_object;
    disk->object_from_render = object_from_render;
    disk->reverse_orientation = reverse_orientation;
    disk->transform_wwapsHandedness = render_from_object.swaps_handedness();

    disk->height = parameters.get_float("height", 0);
    disk->radius = parameters.get_float("radius", 1);
    disk->inner_radius = parameters.get_float("innerradius", 0);

    auto _phi_max = parameters.get_float("phimax", 360);
    disk->phi_max = degree_to_radian(clamp<Real>(_phi_max, 0, 360));

    return disk;
}

PBRT_CPU_GPU
Real Disk::pdf(const ShapeSampleContext &ctx, const Vector3f &wi) const {
    // Intersect sample ray with shape geometry
    Ray ray = ctx.spawn_ray(wi);

    auto isect = intersect(ray);
    if (!isect) {
        return 0;
    }

    // Compute PDF in solid angle measure from shape intersection point
    Real pdf = (1.0 / area()) / (isect->interaction.n.abs_dot(-wi) /
                                      ctx.p().squared_distance(isect->interaction.p()));

    if (isinf(pdf)) {
        pdf = 0;
    }

    return pdf;
}

PBRT_CPU_GPU
pbrt::optional<ShapeSample> Disk::sample(const Point2f &u) const {
    Point2f pd = sample_uniform_disk_concentric(u);
    Point3f pObj(pd.x * radius, pd.y * radius, height);

    Point3fi pi = render_from_object(Point3fi(pObj));

    Normal3f n = render_from_object(Normal3f(0, 0, 1)).normalize();

    if (reverse_orientation) {
        n *= -1;
    }

    // Compute $(u,v)$ for sampled point on disk
    Real phi = std::atan2(pd.y, pd.x);
    if (phi < 0) {
        phi += 2 * pbrt::PI;
    }

    Real radiusSample = std::sqrt(sqr(pObj.x) + sqr(pObj.y));
    Point2f uv(phi / phi_max, (radius - radiusSample) / (radius - inner_radius));

    return ShapeSample{Interaction(pi, n, uv), Real(1.0 / this->area())};
}

PBRT_CPU_GPU
pbrt::optional<ShapeSample> Disk::sample(const ShapeSampleContext &ctx, const Point2f &u) const {
    // Sample shape by area and compute incident direction _wi_
    auto ss = this->sample(u);
    Vector3f wi = ss->interaction.p() - ctx.p();

    if (wi.squared_length() == 0) {
        return {};
    }

    wi = wi.normalize();

    // Convert area sampling PDF in _ss_ to solid angle measure
    ss->pdf /= ss->interaction.n.abs_dot(-wi) / ctx.p().squared_distance(ss->interaction.p());

    if (is_inf(ss->pdf)) {
        return {};
    }

    return ss;
}

PBRT_CPU_GPU
pbrt::optional<QuadricIntersection> Disk::basic_intersect(const Ray &r, Real tMax) const {
    // Transform _Ray_ origin and direction to object space
    Point3fi oi = object_from_render(Point3fi(r.o));
    Vector3fi di = object_from_render(Vector3fi(r.d));

    // Compute plane intersection for disk
    // Reject disk intersections for rays parallel to the disk's plane
    if (Real(di.z) == 0) {
        return {};
    }

    Real tShapeHit = (height - Real(oi.z)) / Real(di.z);
    if (tShapeHit <= 0 || tShapeHit >= tMax) {
        return {};
    }

    // See if hit point is inside disk radii and $\phimax$
    Point3f pHit = oi.to_point3f() + (Real)tShapeHit * di.to_vector3f();

    Real dist2 = sqr(pHit.x) + sqr(pHit.y);
    if (dist2 > sqr(radius) || dist2 < sqr(inner_radius)) {
        return {};
    }

    // Test disk $\phi$ value against $\phimax$
    Real phi = std::atan2(pHit.y, pHit.x);
    if (phi < 0) {
        phi += 2 * pbrt::PI;
    }

    if (phi > phi_max) {
        return {};
    }

    // Return _QuadricIntersection_ for disk intersection
    return QuadricIntersection{tShapeHit, pHit, phi};
}

PBRT_CPU_GPU
SurfaceInteraction Disk::interaction_from_intersection(const QuadricIntersection &isect,
                                                       const Vector3f &wo) const {
    Point3f pHit = isect.p_obj;
    Real phi = isect.phi;
    // Find parametric representation of disk hit
    Real u = phi / phi_max;
    Real rHit = std::sqrt(sqr(pHit.x) + sqr(pHit.y));
    Real v = (radius - rHit) / (radius - inner_radius);
    Vector3f dpdu(-phi_max * pHit.y, phi_max * pHit.x, 0);
    Vector3f dpdv = Vector3f(pHit.x, pHit.y, 0) * (inner_radius - radius) / rHit;
    Normal3f dndu(0, 0, 0), dndv(0, 0, 0);

    // Refine disk intersection point
    pHit.z = height;

    // Compute error bounds for disk intersection
    Vector3f pError(0, 0, 0);

    // Return _SurfaceInteraction_ for quadric intersection
    bool flipNormal = reverse_orientation ^ transform_wwapsHandedness;
    Vector3f woObject = object_from_render(wo);

    return render_from_object(SurfaceInteraction(Point3fi(pHit, pError), Point2f(u, v), woObject,
                                                 dpdu, dpdv, dndu, dndv, flipNormal));
}

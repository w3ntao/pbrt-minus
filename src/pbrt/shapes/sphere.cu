#include "pbrt/shapes/sphere.h"

#include "pbrt/euclidean_space/transform.h"
#include "pbrt/scene/parameter_dict.h"
#include "pbrt/util/util.h"

const Sphere *Sphere::create(const Transform &render_from_object,
                             const Transform &object_from_render, bool reverse_orientation,
                             const ParameterDict &parameters,
                             std::vector<void *> &gpu_dynamic_pointers) {
    auto radius = parameters.get_float("radius", 1.0);
    auto z_min = parameters.get_float("zmin", -radius);
    auto z_max = parameters.get_float("zmax", radius);
    auto phi_max = parameters.get_float("phimax", 360.0);

    Sphere *sphere;
    CHECK_CUDA_ERROR(cudaMallocManaged(&sphere, sizeof(Sphere)));
    gpu_dynamic_pointers.push_back(sphere);

    sphere->init(render_from_object, object_from_render, reverse_orientation, radius, z_min, z_max,
                 phi_max);

    return sphere;
}

void Sphere::init(const Transform &_render_from_object, const Transform &_object_from_render,
                  bool _reverse_orientation, FloatType _radius, FloatType _z_min, FloatType _z_max,
                  FloatType _phi_max) {
    render_from_object = _render_from_object;
    object_from_render = _object_from_render;
    reverse_orientation = _reverse_orientation;

    radius = _radius;
    z_min = clamp(std::min(_z_min, _z_max), -radius, radius);
    z_max = clamp(std::max(_z_min, _z_max), -radius, radius);

    theta_z_min = std::acos(clamp<FloatType>(z_min / radius, -1, 1));
    theta_z_max = std::acos(clamp<FloatType>(z_max / radius, -1, 1));

    phi_max = degree_to_radian(clamp<FloatType>(_phi_max, 0, 360));
}

PBRT_CPU_GPU
Bounds3f Sphere::bounds() const {
    return render_from_object(
        Bounds3f(Point3f(-radius, -radius, z_min), Point3f(radius, radius, z_max)));
}

PBRT_CPU_GPU
cuda::std::optional<QuadricIntersection> Sphere::basic_intersect(const Ray &r,
                                                                 FloatType tMax) const {
    FloatType phi;
    Point3f pHit;
    // Transform _Ray_ origin and direction to object space
    Point3fi oi = object_from_render(Point3fi(r.o));
    Vector3fi di = object_from_render(Vector3fi(r.d));

    // Solve quadratic equation to compute sphere _t0_ and _t1_
    Interval t0, t1;
    // Compute sphere quadratic coefficients
    Interval a = sqr(di.x) + sqr(di.y) + sqr(di.z);
    Interval b = 2 * (di.x * oi.x + di.y * oi.y + di.z * oi.z);
    Interval c = sqr(oi.x) + sqr(oi.y) + sqr(oi.z) - sqr(Interval(radius));

    // Compute sphere quadratic discriminant _discrim_
    Vector3fi v(oi - b / (2 * a) * di);

    Interval length = v.length();
    Interval discrim = 4 * a * (Interval(radius) + length) * (Interval(radius) - length);
    if (discrim.low < 0) {
        return {};
    }

    // Compute quadratic $t$ values
    Interval rootDiscrim = discrim.sqrt();
    Interval q;

    if ((FloatType)b < 0) {
        q = -.5f * (b - rootDiscrim);
    } else {
        q = -.5f * (b + rootDiscrim);
    }

    t0 = q / a;
    t1 = c / q;
    // Swap quadratic $t$ values so that _t0_ is the lesser
    if (t0.low > t1.low) {
        pstd::swap(t0, t1);
    }

    // Check quadric shape _t0_ and _t1_ for nearest intersection
    if (t0.high > tMax || t1.low <= 0) {
        return {};
    }

    Interval tShapeHit = t0;
    if (tShapeHit.low <= 0) {
        tShapeHit = t1;
        if (tShapeHit.high > tMax) {
            return {};
        }
    }

    // Compute sphere hit position and $\phi$
    pHit = oi.to_point3f() + (FloatType)tShapeHit * di.to_vector3f();

    // Refine sphere intersection point
    pHit *= radius / pHit.distance(Point3f(0, 0, 0));

    if (pHit.x == 0 && pHit.y == 0) {
        pHit.x = 1e-5f * radius;
    }

    phi = std::atan2(pHit.y, pHit.x);
    if (phi < 0) {
        phi += 2 * compute_pi();
    }

    // Test sphere intersection against clipping parameters
    if ((z_min > -radius && pHit.z < z_min) || (z_max < radius && pHit.z > z_max) ||
        phi > phi_max) {
        if (tShapeHit == t1) {
            return {};
        }

        if (t1.high > tMax) {
            return {};
        }

        tShapeHit = t1;
        // Compute sphere hit position and $\phi$
        pHit = oi.to_point3f() + (FloatType)tShapeHit * di.to_vector3f();

        // Refine sphere intersection point
        pHit *= radius / pHit.distance(Point3f(0, 0, 0));

        if (pHit.x == 0 && pHit.y == 0) {
            pHit.x = 1e-5f * radius;
        }

        phi = std::atan2(pHit.y, pHit.x);
        if (phi < 0) {
            phi += 2 * compute_pi();
        }

        if ((z_min > -radius && pHit.z < z_min) || (z_max < radius && pHit.z > z_max) ||
            phi > phi_max) {
            return {};
        }
    }

    return QuadricIntersection{FloatType(tShapeHit), pHit, phi};
}

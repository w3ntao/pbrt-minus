#include "pbrt/shapes/sphere.h"

#include "pbrt/base/shape.h"
#include "pbrt/euclidean_space/transform.h"
#include "pbrt/scene/parameter_dictionary.h"

#include "pbrt/util/sampling.h"
#include "pbrt/util/util.h"

const Sphere *Sphere::create(const Transform &render_from_object,
                             const Transform &object_from_render, bool reverse_orientation,
                             const ParameterDictionary &parameters,
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

PBRT_GPU
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

PBRT_GPU
cuda::std::optional<ShapeSample> Sphere::sample(const Point2f &u) const {
    Point3f pObj = Point3f(0, 0, 0) + radius * sample_uniform_sphere(u);

    // Reproject _pObj_ to sphere surface and compute _pObjError_
    pObj *= radius / pObj.distance(Point3f(0, 0, 0));

    Vector3f pObjError = gamma(5) * pObj.to_vector3().abs();

    // Compute surface normal for sphere sample and return _ShapeSample_
    Normal3f nObj(pObj.x, pObj.y, pObj.z);
    Normal3f n = render_from_object(nObj).normalize();

    if (reverse_orientation) {
        n *= -1;
    }

    // Compute $(u, v)$ coordinates for sphere sample
    FloatType theta = safe_acos(pObj.z / radius);
    FloatType phi = std::atan2(pObj.y, pObj.x);
    if (phi < 0) {
        phi += 2 * compute_pi();
    }

    Point2f uv(phi / phi_max, (theta - theta_z_min) / (theta_z_max - theta_z_min));

    Point3fi pi = render_from_object(Point3fi(pObj, pObjError));

    return ShapeSample{.interaction = Interaction(pi, n, uv), .pdf = FloatType(1) / area()};
}

PBRT_GPU
cuda::std::optional<ShapeSample> Sphere::sample(const ShapeSampleContext &ctx,
                                                const Point2f &u) const {
    // Sample uniformly on sphere if $\pt{}$ is inside it
    Point3f pCenter = render_from_object(Point3f(0, 0, 0));
    Point3f pOrigin = ctx.offset_ray_origin(pCenter);

    if (pOrigin.squared_distance(pCenter) <= sqr(radius)) {
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

    // Sample sphere uniformly inside subtended cone
    // Compute quantities related to the $\theta_\roman{max}$ for cone
    FloatType sinThetaMax = radius / ctx.p().distance(pCenter);

    FloatType sin2ThetaMax = sqr(sinThetaMax);
    FloatType cosThetaMax = safe_sqrt(1 - sin2ThetaMax);
    FloatType oneMinusCosThetaMax = 1 - cosThetaMax;

    // Compute $\theta$ and $\phi$ values for sample in cone
    FloatType cosTheta = (cosThetaMax - 1) * u[0] + 1;
    FloatType sin2Theta = 1 - sqr(cosTheta);
    if (sin2ThetaMax < 0.00068523f /* sin^2(1.5 deg) */) {
        // Compute cone sample via Taylor series expansion for small angles
        sin2Theta = sin2ThetaMax * u[0];
        cosTheta = std::sqrt(1 - sin2Theta);
        oneMinusCosThetaMax = sin2ThetaMax / 2;
    }

    // Compute angle $\alpha$ from center of sphere to sampled point on surface
    FloatType cosAlpha =
        sin2Theta / sinThetaMax + cosTheta * safe_sqrt(1 - sin2Theta / sqr(sinThetaMax));
    FloatType sinAlpha = safe_sqrt(1 - sqr(cosAlpha));

    // Compute surface normal and sampled point on sphere
    FloatType phi = u[1] * 2 * compute_pi();
    Vector3f w = SphericalDirection(sinAlpha, cosAlpha, phi);

    Frame samplingFrame = Frame::from_z((pCenter - ctx.p()).normalize());

    Normal3f n(samplingFrame.from_local(-w));

    Point3f p = pCenter + radius * Point3f(n.x, n.y, n.z);
    if (reverse_orientation) {
        n *= -1;
    }

    // Return _ShapeSample_ for sampled point on sphere
    // Compute _pError_ for sampled point on sphere
    Vector3f pError = gamma(5) * p.to_vector3().abs();

    // Compute $(u,v)$ coordinates for sampled point on sphere
    Point3f pObj = object_from_render(p);

    FloatType theta = safe_acos(pObj.z / radius);
    FloatType spherePhi = std::atan2(pObj.y, pObj.x);

    if (spherePhi < 0) {
        spherePhi += 2 * compute_pi();
    }

    Point2f uv(spherePhi / phi_max, (theta - theta_z_min) / (theta_z_max - theta_z_min));

    return ShapeSample{.interaction = Interaction(Point3fi(p, pError), n, uv),
                       .pdf = 1 / (2 * compute_pi() * oneMinusCosThetaMax)};
}

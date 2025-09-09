#include <pbrt/base/shape.h>
#include <pbrt/euclidean_space/transform.h>
#include <pbrt/gpu/gpu_memory_allocator.h>
#include <pbrt/scene/parameter_dictionary.h>
#include <pbrt/shapes/sphere.h>
#include <pbrt/util/sampling.h>

Sphere::Sphere(const Transform &_render_from_object, const Transform &_object_from_render,
               bool _reverse_orientation, const ParameterDictionary &parameters)
    : render_from_object(_render_from_object), object_from_render(_object_from_render),
      reverse_orientation(_reverse_orientation),
      transform_swaps_handedness(render_from_object.swaps_handedness()),
      radius(parameters.get_float("radius", 1.0)) {

    const auto _z_min = parameters.get_float("zmin", -radius);
    const auto _z_max = parameters.get_float("zmax", radius);
    const auto _phi_max = parameters.get_float("phimax", 360.0);

    z_min = clamp(std::min(_z_min, _z_max), -radius, radius);
    z_max = clamp(std::max(_z_min, _z_max), -radius, radius);

    theta_z_min = std::acos(clamp<Real>(z_min / radius, -1, 1));
    theta_z_max = std::acos(clamp<Real>(z_max / radius, -1, 1));

    phi_max = degree_to_radian(clamp<Real>(_phi_max, 0, 360));
}

PBRT_CPU_GPU
Bounds3f Sphere::bounds() const {
    return render_from_object(
        Bounds3f(Point3f(-radius, -radius, z_min), Point3f(radius, radius, z_max)));
}

PBRT_CPU_GPU
pbrt::optional<QuadricIntersection> Sphere::basic_intersect(const Ray &r, const Real tMax) const {
    // Transform _Ray_ origin and direction to object space
    Point3fi oi = object_from_render(Point3fi(r.o));
    Vector3fi di = object_from_render(Vector3fi(r.d));

    // Compute sphere quadratic coefficients
    Interval a = sqr(di.x) + sqr(di.y) + sqr(di.z);
    Interval b = 2 * (di.x * oi.x + di.y * oi.y + di.z * oi.z);
    Interval c = sqr(oi.x) + sqr(oi.y) + sqr(oi.z) - sqr(Interval(radius));

    // Compute sphere quadratic discriminant _discrim_
    const auto v = Vector3fi(oi - b / (2 * a) * di);

    Interval length = v.length();
    Interval discrim = 4 * a * (Interval(radius) + length) * (Interval(radius) - length);
    if (discrim.low < 0) {
        return {};
    }

    // Compute quadratic $t$ values
    const Interval rootDiscrim = discrim.sqrt();
    const Interval q = (Real)b < 0 ? -.5f * (b - rootDiscrim) : -.5f * (b + rootDiscrim);

    // Solve quadratic equation to compute sphere _t0_ and _t1_
    Interval t0 = q / a;

    Interval t1 = c / q;
    // Swap quadratic $t$ values so that _t0_ is the lesser
    if (t0.low > t1.low) {
        pbrt::swap(t0, t1);
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
    auto pHit = oi.to_point3f() + (Real)tShapeHit * di.to_vector3f();

    // Refine sphere intersection point
    pHit *= radius / pHit.distance(Point3f(0, 0, 0));

    if (pHit.x == 0 && pHit.y == 0) {
        pHit.x = 1e-5f * radius;
    }

    auto phi = std::atan2(pHit.y, pHit.x);
    if (phi < 0) {
        phi += 2 * pbrt::PI;
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
        pHit = oi.to_point3f() + (Real)tShapeHit * di.to_vector3f();

        // Refine sphere intersection point
        pHit *= radius / pHit.distance(Point3f(0, 0, 0));

        if (pHit.x == 0 && pHit.y == 0) {
            pHit.x = 1e-5f * radius;
        }

        phi = std::atan2(pHit.y, pHit.x);
        if (phi < 0) {
            phi += 2 * pbrt::PI;
        }

        if ((z_min > -radius && pHit.z < z_min) || (z_max < radius && pHit.z > z_max) ||
            phi > phi_max) {
            return {};
        }
    }

    return QuadricIntersection{Real(tShapeHit), pHit, phi};
}

PBRT_CPU_GPU
pbrt::optional<ShapeSample> Sphere::sample(const Point2f &u) const {
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
    Real theta = safe_acos(pObj.z / radius);
    Real phi = std::atan2(pObj.y, pObj.x);
    if (phi < 0) {
        phi += 2 * pbrt::PI;
    }

    Point2f uv(phi / phi_max, (theta - theta_z_min) / (theta_z_max - theta_z_min));

    Point3fi pi = render_from_object(Point3fi(pObj, pObjError));

    return ShapeSample{.interaction = Interaction(pi, n, uv), .pdf = Real(1) / area()};
}

PBRT_CPU_GPU
pbrt::optional<ShapeSample> Sphere::sample(const ShapeSampleContext &ctx, const Point2f &u) const {
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
        ss->pdf *= ctx.p().squared_distance(ss->interaction.p()) / ss->interaction.n.abs_dot(-wi);

        if (is_inf(ss->pdf)) {
            return {};
        }

        return ss;
    }

    // Sample sphere uniformly inside subtended cone
    // Compute quantities related to the $\theta_\roman{max}$ for cone
    Real sinThetaMax = radius / ctx.p().distance(pCenter);

    Real sin2ThetaMax = sqr(sinThetaMax);
    Real cosThetaMax = safe_sqrt(1 - sin2ThetaMax);
    Real oneMinusCosThetaMax = 1 - cosThetaMax;

    // Compute $\theta$ and $\phi$ values for sample in cone
    Real cosTheta = (cosThetaMax - 1) * u[0] + 1;
    Real sin2Theta = 1 - sqr(cosTheta);
    if (sin2ThetaMax < 0.00068523f /* sin^2(1.5 deg) */) {
        // Compute cone sample via Taylor series expansion for small angles
        sin2Theta = sin2ThetaMax * u[0];
        cosTheta = std::sqrt(1 - sin2Theta);
        oneMinusCosThetaMax = sin2ThetaMax / 2;
    }

    // Compute angle $\alpha$ from center of sphere to sampled point on surface
    Real cosAlpha =
        sin2Theta / sinThetaMax + cosTheta * safe_sqrt(1 - sin2Theta / sqr(sinThetaMax));
    Real sinAlpha = safe_sqrt(1 - sqr(cosAlpha));

    // Compute surface normal and sampled point on sphere
    Real phi = u[1] * 2 * pbrt::PI;
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

    Real theta = safe_acos(pObj.z / radius);
    Real spherePhi = std::atan2(pObj.y, pObj.x);

    if (spherePhi < 0) {
        spherePhi += 2 * pbrt::PI;
    }

    Point2f uv(spherePhi / phi_max, (theta - theta_z_min) / (theta_z_max - theta_z_min));

    return ShapeSample{.interaction = Interaction(Point3fi(p, pError), n, uv),
                       .pdf = 1 / (2 * pbrt::PI * oneMinusCosThetaMax)};
}

PBRT_CPU_GPU
Real Sphere::pdf(const ShapeSampleContext &ctx, const Vector3f &wi) const {
    Point3f pCenter = render_from_object(Point3f(0, 0, 0));
    Point3f pOrigin = ctx.offset_ray_origin(pCenter);

    if (pOrigin.squared_distance(pCenter) <= sqr(radius)) {
        // Return solid angle PDF for point inside sphere
        // Intersect sample ray with shape geometry
        Ray ray = ctx.spawn_ray(wi);
        auto isect = this->intersect(ray);
        if (!isect) {
            return 0;
        }

        // Compute PDF in solid angle measure from shape intersection point
        Real pdf = ctx.p().squared_distance(isect->interaction.p()) /
                   (area() * isect->interaction.n.abs_dot(-wi));

        if (isinf(pdf)) {
            pdf = 0;
        }

        return pdf;
    }

    // Compute general solid angle sphere PDF
    Real sin2ThetaMax = radius * radius / ctx.p().squared_distance(pCenter);
    Real cosThetaMax = safe_sqrt(1 - sin2ThetaMax);
    Real oneMinusCosThetaMax = 1 - cosThetaMax;

    // Compute more accurate _oneMinusCosThetaMax_ for small solid angle
    if (sin2ThetaMax < 0.00068523f /* sin^2(1.5 deg) */) {
        oneMinusCosThetaMax = sin2ThetaMax / 2;
    }

    return 1.0 / (2.0 * pbrt::PI * oneMinusCosThetaMax);
}

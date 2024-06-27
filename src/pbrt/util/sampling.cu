#include "pbrt/util/sampling.h"
#include "pbrt/base/media.h"
#include "pbrt/euclidean_space/frame.h"

PBRT_CPU_GPU
Vector3f SampleHenyeyGreenstein(Vector3f wo, FloatType g, Point2f u, FloatType *pdf) {
    // When g \approx -1 and u[0] \approx 0 or with g \approx 1 and u[0]
    // \approx 1, the computation of cosTheta below is unstable and can
    // give, leading to NaNs. For now we limit g to the range where it is
    // still ok; it would be nice to come up with a numerically robust
    // rewrite of the computation instead, but with |g| \approx 1, the
    // sampling distribution has a very sharp turn to deal with.
    g = clamp<FloatType>(g, -.99, .99);

    // Compute $\cos\theta$ for Henyey--Greenstein sample
    FloatType cosTheta;
    if (std::abs(g) < 1e-3f) {
        cosTheta = 1 - 2 * u[0];
    } else {
        cosTheta = -1 / (2 * g) * (1 + sqr(g) - sqr((1 - sqr(g)) / (1 + g - 2 * g * u[0])));
    }

    // Compute direction _wi_ for Henyey--Greenstein sample
    FloatType sinTheta = safe_sqrt(1 - sqr(cosTheta));
    FloatType phi = 2 * compute_pi() * u[1];

    Frame wFrame = Frame::from_z(wo);

    // Vector3f wi = wFrame.FromLocal(SphericalDirection(sinTheta, cosTheta, phi));
    Vector3f wi = wFrame.from_local(SphericalDirection(sinTheta, cosTheta, phi));

    if (pdf) {
        *pdf = HenyeyGreenstein(cosTheta, g);
    }

    return wi;
}

PBRT_CPU_GPU
// Via Jim Arvo's SphTri.C
Point2f InvertSphericalTriangleSample(const Point3f v[3], const Point3f &p, const Vector3f &w) {
    // Compute vectors _a_, _b_, and _c_ to spherical triangle vertices
    Vector3f a(v[0] - p);
    Vector3f b(v[1] - p);
    Vector3f c(v[2] - p);

    a = a.normalize();
    b = b.normalize();
    c = c.normalize();

    // Compute normalized cross products of all direction pairs
    Vector3f n_ab = a.cross(b);
    Vector3f n_bc = b.cross(c);
    Vector3f n_ca = c.cross(a);

    if (n_ab.squared_length() == 0 || n_bc.squared_length() == 0 || n_ca.squared_length() == 0) {
        return {};
    }

    n_ab = n_ab.normalize();
    n_bc = n_bc.normalize();
    n_ca = n_ca.normalize();

    // Find angles $\alpha$, $\beta$, and $\gamma$ at spherical triangle vertices
    FloatType alpha = angle_between(n_ab, -n_ca);
    FloatType beta = angle_between(n_bc, -n_ab);
    FloatType gamma = angle_between(n_ca, -n_bc);

    // Find vertex $\VEC{c'}$ along $\VEC{a}\VEC{c}$ arc for $\w{}$
    Vector3f cp = b.cross(w).cross(c.cross(a)).normalize();
    if (cp.dot(a + c) < 0) {
        cp = -cp;
    }

    // Invert uniform area sampling to find _u0_
    FloatType u0;
    if (a.dot(cp) > 0.99999847691f /* 0.1 degrees */) {
        u0 = 0;
    } else {
        // Compute area $A'$ of subtriangle
        Vector3f n_cpb = cp.cross(b);
        Vector3f n_acp = a.cross(cp);

        if (n_cpb.squared_length() == 0 || n_acp.squared_length() == 0) {
            return Point2f(0.5, 0.5);
        }

        n_cpb = n_cpb.normalize();
        n_acp = n_acp.normalize();
        FloatType Ap =
            alpha + angle_between(n_ab, n_cpb) + angle_between(n_acp, -n_cpb) - compute_pi();

        // Compute sample _u0_ that gives the area $A'$
        FloatType A = alpha + beta + gamma - compute_pi();
        u0 = Ap / A;
    }

    // Invert arc sampling to find _u1_ and return result
    FloatType u1 = (1 - w.dot(b)) / (1 - cp.dot(b));
    return Point2f(clamp<FloatType>(u0, 0, 1), clamp<FloatType>(u1, 0, 1));
}
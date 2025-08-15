#include <pbrt/euclidean_space/frame.h>
#include <pbrt/medium/media_util.h>
#include <pbrt/util/math.h>
#include <pbrt/util/sampling.h>

PBRT_CPU_GPU
Vector3f sample_henyey_greenstein(const Vector3f &wo, Real g, const Point2f u, Real *pdf) {
    // When g \approx -1 and u[0] \approx 0 or with g \approx 1 and u[0]
    // \approx 1, the computation of cosTheta below is unstable and can
    // give, leading to NaNs. For now we limit g to the range where it is
    // still ok; it would be nice to come up with a numerically robust
    // rewrite of the computation instead, but with |g| \approx 1, the
    // sampling distribution has a very sharp turn to deal with.
    g = clamp<Real>(g, -.99, .99);

    // Compute $\cos\theta$ for Henyey--Greenstein sample
    Real cosTheta;
    if (std::abs(g) < 1e-3f) {
        cosTheta = 1 - 2 * u[0];
    } else {
        cosTheta = -1 / (2 * g) * (1 + sqr(g) - sqr((1 - sqr(g)) / (1 + g - 2 * g * u[0])));
    }

    // Compute direction _wi_ for Henyey--Greenstein sample
    Real sinTheta = safe_sqrt(1 - sqr(cosTheta));
    Real phi = 2 * pbrt::PI * u[1];

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
    Real alpha = angle_between(n_ab, -n_ca);
    Real beta = angle_between(n_bc, -n_ab);
    Real gamma = angle_between(n_ca, -n_bc);

    // Find vertex $\VEC{c'}$ along $\VEC{a}\VEC{c}$ arc for $\w{}$
    Vector3f cp = b.cross(w).cross(c.cross(a)).normalize();
    if (cp.dot(a + c) < 0) {
        cp = -cp;
    }

    // Invert uniform area sampling to find _u0_
    Real u0;
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
        Real Ap = alpha + angle_between(n_ab, n_cpb) + angle_between(n_acp, -n_cpb) - pbrt::PI;

        // Compute sample _u0_ that gives the area $A'$
        Real A = alpha + beta + gamma - pbrt::PI;
        u0 = Ap / A;
    }

    // Invert arc sampling to find _u1_ and return result
    Real u1 = (1 - w.dot(b)) / (1 - cp.dot(b));
    return Point2f(clamp<Real>(u0, 0, 1), clamp<Real>(u1, 0, 1));
}

PBRT_CPU_GPU
Point2f EqualAreaSphereToSquare(Vector3f d) {
    Real x = std::abs(d.x);
    Real y = std::abs(d.y);
    Real z = std::abs(d.z);

    // Compute the radius r
    Real r = safe_sqrt(1 - z); // r = sqrt(1-|z|)

    // Compute the argument to atan (detect a=0 to avoid div-by-zero)
    Real a = std::max(x, y);
    Real b = std::min(x, y);
    b = a == 0 ? 0 : b / a;

    // Polynomial approximation of atan(x)*2/pi, x=b
    // Coefficients for 6th degree minimax approximation of atan(x)*2/pi,
    // x=[0,1].
    const Real t1 = 0.406758566246788489601959989e-5;
    const Real t2 = 0.636226545274016134946890922156;
    const Real t3 = 0.61572017898280213493197203466e-2;
    const Real t4 = -0.247333733281268944196501420480;
    const Real t5 = 0.881770664775316294736387951347e-1;
    const Real t6 = 0.419038818029165735901852432784e-1;
    const Real t7 = -0.251390972343483509333252996350e-1;
    Real phi = evaluate_polynomial(b, t1, t2, t3, t4, t5, t6, t7);

    // Extend phi if the input is in the range 45-90 degrees (u<v)
    if (x < y)
        phi = 1 - phi;

    // Find (u,v) based on (r,phi)
    Real v = phi * r;
    Real u = r - v;

    if (d.z < 0) {
        // southern hemisphere -> mirror u,v
        pbrt::swap(u, v);
        u = 1 - u;
        v = 1 - v;
    }

    // Move (u,v) to the correct quadrant based on the signs of (x,y)
    u = pbrt::copysign(u, d.x);
    v = pbrt::copysign(v, d.y);

    // Transform (u,v) from [-1,1] to [0,1]
    return Point2f(0.5f * (u + 1), 0.5f * (v + 1));
}

// Square--Sphere Mapping Function Definitions
// Via source code from Clarberg: Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD
PBRT_CPU_GPU
Vector3f EqualAreaSquareToSphere(Point2f p) {
    // Transform _p_ to $[-1,1]^2$ and compute absolute values
    Real u = 2 * p.x - 1, v = 2 * p.y - 1;
    Real up = std::abs(u), vp = std::abs(v);

    // Compute radius _r_ as signed distance from diagonal
    Real signedDistance = 1 - (up + vp);
    Real d = std::abs(signedDistance);
    Real r = 1 - d;

    // Compute angle $\phi$ for square to sphere mapping
    Real phi = (r == 0 ? 1 : (vp - up) / r + 1) * pbrt::PI / 4;

    // Find $z$ coordinate for spherical direction
    Real z = pbrt::copysign(1 - sqr(r), signedDistance);

    // Compute $\cos\phi$ and $\sin\phi$ for original quadrant and return vector
    Real cosPhi = pbrt::copysign(std::cos(phi), u);
    Real sinPhi = pbrt::copysign(std::sin(phi), v);
    return Vector3f(cosPhi * r * safe_sqrt(2 - sqr(r)), sinPhi * r * safe_sqrt(2 - sqr(r)), z);
}

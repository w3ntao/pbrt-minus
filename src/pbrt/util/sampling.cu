#include <pbrt/base/media.h>
#include <pbrt/euclidean_space/frame.h>
#include <pbrt/util/basic_math.h>
#include <pbrt/util/sampling.h>
#include <pbrt/util/util.h>

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

PBRT_CPU_GPU
Point2f EqualAreaSphereToSquare(Vector3f d) {
    FloatType x = std::abs(d.x);
    FloatType y = std::abs(d.y);
    FloatType z = std::abs(d.z);

    // Compute the radius r
    FloatType r = safe_sqrt(1 - z); // r = sqrt(1-|z|)

    // Compute the argument to atan (detect a=0 to avoid div-by-zero)
    FloatType a = std::max(x, y);
    FloatType b = std::min(x, y);
    b = a == 0 ? 0 : b / a;

    // Polynomial approximation of atan(x)*2/pi, x=b
    // Coefficients for 6th degree minimax approximation of atan(x)*2/pi,
    // x=[0,1].
    const FloatType t1 = 0.406758566246788489601959989e-5;
    const FloatType t2 = 0.636226545274016134946890922156;
    const FloatType t3 = 0.61572017898280213493197203466e-2;
    const FloatType t4 = -0.247333733281268944196501420480;
    const FloatType t5 = 0.881770664775316294736387951347e-1;
    const FloatType t6 = 0.419038818029165735901852432784e-1;
    const FloatType t7 = -0.251390972343483509333252996350e-1;
    FloatType phi = evaluate_polynomial(b, t1, t2, t3, t4, t5, t6, t7);

    // Extend phi if the input is in the range 45-90 degrees (u<v)
    if (x < y)
        phi = 1 - phi;

    // Find (u,v) based on (r,phi)
    FloatType v = phi * r;
    FloatType u = r - v;

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
    FloatType u = 2 * p.x - 1, v = 2 * p.y - 1;
    FloatType up = std::abs(u), vp = std::abs(v);

    // Compute radius _r_ as signed distance from diagonal
    FloatType signedDistance = 1 - (up + vp);
    FloatType d = std::abs(signedDistance);
    FloatType r = 1 - d;

    // Compute angle $\phi$ for square to sphere mapping
    FloatType phi = (r == 0 ? 1 : (vp - up) / r + 1) * compute_pi() / 4;

    // Find $z$ coordinate for spherical direction
    FloatType z = pbrt::copysign(1 - sqr(r), signedDistance);

    // Compute $\cos\phi$ and $\sin\phi$ for original quadrant and return vector
    FloatType cosPhi = pbrt::copysign(std::cos(phi), u);
    FloatType sinPhi = pbrt::copysign(std::sin(phi), v);
    return Vector3f(cosPhi * r * safe_sqrt(2 - sqr(r)), sinPhi * r * safe_sqrt(2 - sqr(r)), z);
}

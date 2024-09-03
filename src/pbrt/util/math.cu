#include "pbrt/util/basic_math.h"
#include "pbrt/util/math.h"
#include "pbrt/util/util.h"

PBRT_GPU
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
        pstd::swap(u, v);
        u = 1 - u;
        v = 1 - v;
    }

    // Move (u,v) to the correct quadrant based on the signs of (x,y)
    u = pstd::copysign(u, d.x);
    v = pstd::copysign(v, d.y);

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
    FloatType z = pstd::copysign(1 - sqr(r), signedDistance);

    // Compute $\cos\phi$ and $\sin\phi$ for original quadrant and return vector
    FloatType cosPhi = pstd::copysign(std::cos(phi), u);
    FloatType sinPhi = pstd::copysign(std::sin(phi), v);
    return Vector3f(cosPhi * r * safe_sqrt(2 - sqr(r)), sinPhi * r * safe_sqrt(2 - sqr(r)), z);
}

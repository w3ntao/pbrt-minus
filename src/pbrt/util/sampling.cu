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

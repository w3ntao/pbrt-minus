#pragma once

#include <pbrt/util/optional.h>
#include <pbrt/util/scattering.h>

struct PhaseFunctionSample {
    Real rho;
    Vector3f wi;
    Real pdf;
};

class HGPhaseFunction {
  public:
    PBRT_CPU_GPU
    HGPhaseFunction() {}

    PBRT_CPU_GPU
    HGPhaseFunction(Real _g) : g(_g) {}

    PBRT_CPU_GPU
    Real eval(const Vector3f &wo, const Vector3f &wi) const {
        return HenyeyGreenstein(wo.dot(wi), g);
    }

    PBRT_CPU_GPU
    pbrt::optional<PhaseFunctionSample> sample(const Vector3f &wo, const Point2f &u) const {
        Real pdf;
        Vector3f wi = sample_henyey_greenstein(wo, g, u, &pdf);
        return PhaseFunctionSample{pdf, wi, pdf};
    }

    PBRT_CPU_GPU
    Real pdf(const Vector3f &wo, const Vector3f &wi) const {
        return eval(wo, wi);
    }

  private:
    Real g = NAN;
};

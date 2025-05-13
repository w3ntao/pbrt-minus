#pragma once

#include <pbrt/base/spectrum.h>
#include <pbrt/util/optional.h>
#include <pbrt/util/scattering.h>

struct PhaseFunctionSample {
    Real p;
    Vector3f wi;
    Real pdf;
};

class HGPhaseFunction {
  public:
    PBRT_CPU_GPU
    HGPhaseFunction() {}

    PBRT_CPU_GPU
    HGPhaseFunction(Real g) : g(g) {}

    PBRT_CPU_GPU
    Real p(const Vector3f wo, const Vector3f wi) const {
        return HenyeyGreenstein(wo.dot(wi), g);
    }

    PBRT_CPU_GPU
    pbrt::optional<PhaseFunctionSample> sample_p(Vector3f wo, Point2f u) const {
        Real pdf;
        Vector3f wi = SampleHenyeyGreenstein(wo, g, u, &pdf);
        return PhaseFunctionSample{pdf, wi, pdf};
    }

    PBRT_CPU_GPU
    Real pdf(Vector3f wo, Vector3f wi) const {
        return p(wo, wi);
    }

  private:
    Real g;
};

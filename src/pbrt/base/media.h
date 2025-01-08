#pragma once

#include "pbrt/base/spectrum.h"
#include "pbrt/util/optional.h"
#include "pbrt/util/scattering.h"

struct PhaseFunctionSample {
    FloatType p;
    Vector3f wi;
    FloatType pdf;
};

class HGPhaseFunction {
  public:
    PBRT_GPU
    HGPhaseFunction() {}

    PBRT_GPU
    HGPhaseFunction(FloatType g) : g(g) {}

    PBRT_GPU
    FloatType p(const Vector3f wo, const Vector3f wi) const {
        return HenyeyGreenstein(wo.dot(wi), g);
    }

    PBRT_GPU
    pbrt::optional<PhaseFunctionSample> sample_p(Vector3f wo, Point2f u) const {
        FloatType pdf;
        Vector3f wi = SampleHenyeyGreenstein(wo, g, u, &pdf);
        return PhaseFunctionSample{pdf, wi, pdf};
    }

    PBRT_GPU
    FloatType pdf(Vector3f wo, Vector3f wi) const {
        return p(wo, wi);
    }

  private:
    FloatType g;
};

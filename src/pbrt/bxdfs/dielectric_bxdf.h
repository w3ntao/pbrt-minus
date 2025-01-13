#pragma once

#include <pbrt/base/bxdf_util.h>
#include <pbrt/gpu/macro.h>
#include <pbrt/util/optional.h>
#include <pbrt/util/scattering.h>

class DielectricBxDF {
  public:
    PBRT_CPU_GPU
    DielectricBxDF() : eta(NAN) {}

    PBRT_CPU_GPU
    DielectricBxDF(FloatType eta, TrowbridgeReitzDistribution mfDistrib)
        : eta(eta), mfDistrib(mfDistrib) {}

    PBRT_CPU_GPU
    BxDFFlags flags() const {
        BxDFFlags _flags = eta == 1 ? Transmission : Reflection | Transmission;

        return _flags | (mfDistrib.effectively_smooth() ? Specular : Glossy);
    }

    PBRT_CPU_GPU
    pbrt::optional<BSDFSample>
    sample_f(Vector3f wo, FloatType uc, Point2f u, TransportMode mode,
             BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All) const;

    PBRT_CPU_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const;

    PBRT_CPU_GPU
    FloatType pdf(Vector3f wo, Vector3f wi, TransportMode mode,
                  BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    PBRT_CPU_GPU
    void regularize() {
        mfDistrib.regularize();
    }

  private:
    FloatType eta;
    TrowbridgeReitzDistribution mfDistrib;
};

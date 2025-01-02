#pragma once

#include "pbrt/base/bxdf_util.h"
#include "pbrt/util/macro.h"
#include "pbrt/util/scattering.h"
#include <cuda/std/optional>

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
    cuda::std::optional<BSDFSample>
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

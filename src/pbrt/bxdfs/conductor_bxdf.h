#pragma once

#include "pbrt/base/bxdf.h"
#include "pbrt/util/scattering.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"

class ConductorBxDF {
  public:
    // ConductorBxDF Public Methods
    PBRT_CPU_GPU
    ConductorBxDF() : eta(SampledSpectrum(NAN)), k(SampledSpectrum(NAN)) {}

    PBRT_GPU
    ConductorBxDF(const TrowbridgeReitzDistribution &_mf_distrib, const SampledSpectrum &_eta,
                  const SampledSpectrum &_k)
        : mf_distrib(_mf_distrib), eta(_eta), k(_k) {}

    PBRT_CPU_GPU
    BxDFFlags flags() const {
        return mf_distrib.effectively_smooth() ? BxDFFlags::SpecularReflection
                                               : BxDFFlags::GlossyReflection;
    }

    PBRT_GPU
    cuda::std::optional<BSDFSample>
    sample_f(Vector3f wo, FloatType uc, Point2f u, TransportMode mode,
             BxDFReflTransFlags sample_flags = BxDFReflTransFlags::All);

    PBRT_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const;

    PBRT_GPU
    FloatType pdf(Vector3f wo, Vector3f wi, TransportMode mode,
                  BxDFReflTransFlags sample_flags) const;

    PBRT_GPU
    void regularize() {
        mf_distrib.regularize();
    }

  private:
    // ConductorBxDF Private Members
    TrowbridgeReitzDistribution mf_distrib;
    SampledSpectrum eta;
    SampledSpectrum k;
};

#pragma once

#include <cuda/std/variant>
#include <pbrt/bxdfs/coated_conductor_bxdf.h>
#include <pbrt/bxdfs/coated_diffuse_bxdf.h>
#include <pbrt/bxdfs/conductor_bxdf.h>
#include <pbrt/bxdfs/dielectric_bxdf.h>
#include <pbrt/bxdfs/diffuse_bxdf.h>
#include <pbrt/bxdfs/diffuse_transmission_bxdf.h>

namespace HIDDEN {
using BxDFVariants = cuda::std::variant<CoatedConductorBxDF, CoatedDiffuseBxDF, ConductorBxDF,
                                        DielectricBxDF, DiffuseBxDF, DiffuseTransmissionBxDF>;
}

class BxDF : public HIDDEN::BxDFVariants {
    using HIDDEN::BxDFVariants::BxDFVariants;

  public:
    PBRT_CPU_GPU
    BxDFFlags flags() const {
        return cuda::std::visit([&](auto &x) { return x.flags(); }, *this);
    }

    PBRT_CPU_GPU
    void regularize() {
        cuda::std::visit([&](auto &x) { x.regularize(); }, *this);
    }

    PBRT_CPU_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const {
        return cuda::std::visit([&](auto &x) { return x.f(wo, wi, mode); }, *this);
    }

    PBRT_CPU_GPU
    pbrt::optional<BSDFSample>
    sample_f(Vector3f wo, Real uc, Point2f u, TransportMode mode = TransportMode::Radiance,
             BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        return cuda::std::visit([&](auto &x) { return x.sample_f(wo, uc, u, mode, sampleFlags); },
                                *this);
    }

    PBRT_CPU_GPU
    Real pdf(Vector3f wo, Vector3f wi, TransportMode mode,
             BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const {
        return cuda::std::visit([&](auto &x) { return x.pdf(wo, wi, mode, sampleFlags); }, *this);
    }
};

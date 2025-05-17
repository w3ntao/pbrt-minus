#pragma once

#include <pbrt/base/bxdf_util.h>
#include <pbrt/bxdfs/coated_conductor_bxdf.h>
#include <pbrt/bxdfs/coated_diffuse_bxdf.h>
#include <pbrt/bxdfs/conductor_bxdf.h>
#include <pbrt/bxdfs/dielectric_bxdf.h>
#include <pbrt/bxdfs/diffuse_bxdf.h>
#include <pbrt/bxdfs/diffuse_transmission_bxdf.h>
#include <pbrt/gpu/macro.h>
#include <pbrt/spectrum_util/sampled_spectrum.h>

class BxDF {
  public:
    enum class Type {
        null,
        coated_conductor,
        coated_diffuse,
        conductor,
        dielectric,
        diffuse,
        diffuse_transmission,
    };

    PBRT_CPU_GPU
    BxDF() : type(Type::null) {}

    PBRT_CPU_GPU
    void init(const CoatedConductorBxDF &_coated_conductor_bxdf);

    PBRT_CPU_GPU
    void init(const CoatedDiffuseBxDF &_coated_diffuse_bxdf);

    PBRT_CPU_GPU
    void init(const ConductorBxDF &_conductor_bxdf);

    PBRT_CPU_GPU
    void init(const DielectricBxDF &_dielectric_bxdf);

    PBRT_CPU_GPU
    void init(const DiffuseBxDF &_diffuse_bxdf);

    PBRT_CPU_GPU
    void init(const DiffuseTransmissionBxDF &_diffuse_transmission_bxdf);

    PBRT_CPU_GPU
    BxDFFlags flags() const;

    PBRT_CPU_GPU
    void regularize();

    PBRT_CPU_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const;

    PBRT_CPU_GPU
    pbrt::optional<BSDFSample>
    sample_f(Vector3f wo, Real uc, Point2f u, TransportMode mode = TransportMode::Radiance,
             BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    PBRT_CPU_GPU
    Real pdf(Vector3f wo, Vector3f wi, TransportMode mode,
             BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    PBRT_CPU_GPU
    bool has_type_null() const {
        return type == Type::null;
    }

  private:
    Type type;

    CoatedConductorBxDF coated_conductor_bxdf;
    CoatedDiffuseBxDF coated_diffuse_bxdf;
    ConductorBxDF conductor_bxdf;
    DielectricBxDF dielectric_bxdf;
    DiffuseBxDF diffuse_bxdf;
    DiffuseTransmissionBxDF diffuse_transmission_bxdf;
};

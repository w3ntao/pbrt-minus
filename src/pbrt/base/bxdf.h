#pragma once

#include "pbrt/base/bxdf_util.h"
#include "pbrt/bxdfs/full_bxdf.h"
#include "pbrt/spectrum_util/sampled_spectrum.h"
#include "pbrt/util/macro.h"
#include <cuda/std/optional>

class BxDF {
  public:
    enum class Type {
        null,
        coated_conductor,
        coated_diffuse,
        conductor,
        dielectric,
        diffuse,
    };

    PBRT_GPU
    BxDF() : type(Type::null) {}

    PBRT_GPU
    void init(const CoatedConductorBxDF &_coated_conductor_bxdf);

    PBRT_GPU
    void init(const CoatedDiffuseBxDF &_coated_diffuse_bxdf);

    PBRT_GPU
    void init(const ConductorBxDF &_conductor_bxdf);

    PBRT_GPU
    void init(const DielectricBxDF &_dielectric_bxdf);

    PBRT_GPU
    void init(const DiffuseBxDF &_diffuse_bxdf);

    PBRT_CPU_GPU
    BxDFFlags flags() const;

    PBRT_GPU
    void regularize();

    PBRT_GPU
    SampledSpectrum f(Vector3f wo, Vector3f wi, TransportMode mode) const;

    PBRT_GPU
    cuda::std::optional<BSDFSample>
    sample_f(Vector3f wo, FloatType uc, Point2f u, TransportMode mode = TransportMode::Radiance,
             BxDFReflTransFlags sampleFlags = BxDFReflTransFlags::All) const;

    PBRT_GPU
    FloatType pdf(Vector3f wo, Vector3f wi, TransportMode mode,
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
};
